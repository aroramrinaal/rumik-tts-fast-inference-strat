import torch
import triton
import triton.language as tl

N_LOGITS = 2049


EOS_LOCAL = 2048


BLOCK = 1024


MASK31 = 0x7FFFFFFF


@triton.jit
def _tail_kernel(Logits, PhaseRow, StopIn, NextTok, Hist, Action,
                 emit_pos, step, min_tokens, temperature, seed,
                 K_TOP: tl.constexpr, DO_SELECT: tl.constexpr,
                 DO_SAMPLE: tl.constexpr, BLOCK: tl.constexpr,
                 N: tl.constexpr, EOS: tl.constexpr, M31: tl.constexpr,
                 PhaseOffset=None, DEVICE_CONTROL: tl.constexpr=False):
    if DEVICE_CONTROL:
        emit_pos = tl.load(emit_pos)
        step = tl.load(step)
        min_tokens = tl.load(min_tokens)
        temperature = tl.load(temperature)
        seed = tl.load(seed)
        PhaseRow += tl.load(PhaseOffset) * N
    tid = tl.arange(0, BLOCK)
    i0 = tid
    i1 = tid + BLOCK
    first = tid == 0

    t = tl.maximum(temperature, 1e-5)
    l0 = tl.load(Logits + i0).to(tl.float32)
    l1 = tl.load(Logits + i1).to(tl.float32)
    v0 = (l0 / t).to(tl.bfloat16)
    v1 = (l1 / t).to(tl.bfloat16)
    eos_logit = tl.load(Logits + EOS).to(tl.float32)
    v2 = ((eos_logit / t).to(tl.bfloat16)).to(tl.float32) + tl.zeros([BLOCK], tl.float32)
    v0f = v0.to(tl.float32)
    v1f = v1.to(tl.float32)
    v2raw = v2
    if DO_SAMPLE:

        v2 = tl.where(step < min_tokens,
                      tl.full([BLOCK], -3.38953139e38, tl.float32), v2)

        if DO_SELECT:
            b0 = v0.to(tl.uint16, bitcast=True).to(tl.uint32)
            b1 = v1.to(tl.uint16, bitcast=True).to(tl.uint32)
            b2 = v2.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)

            b0 = tl.where(b0 == 0x8000, 0, b0)
            b1 = tl.where(b1 == 0x8000, 0, b1)
            b2 = tl.where(b2 == 0x8000, 0, b2)
            neg0 = (b0 >> 15) & 1
            neg1 = (b1 >> 15) & 1
            neg2 = (b2 >> 15) & 1
            o0 = tl.where(neg0 == 1, (~b0) & 0xFFFF, b0 ^ 0x8000)
            o1 = tl.where(neg1 == 1, (~b1) & 0xFFFF, b1 ^ 0x8000)
            o2 = tl.where(neg2 == 1, (~b2) & 0xFFFF, b2 ^ 0x8000)
            pref = 0
            kk = N - K_TOP
            for bitpos in tl.static_range(16):
                b = 15 - bitpos

                if b == 15:
                    c0 = ((o0 >> b) & 1) == 0
                    c1 = ((o1 >> b) & 1) == 0
                    c2 = (((o2 >> b) & 1) == 0) & first
                else:
                    sh = b + 1
                    c0 = ((o0 >> sh) == (pref >> sh)) & (((o0 >> b) & 1) == 0)
                    c1 = ((o1 >> sh) == (pref >> sh)) & (((o1 >> b) & 1) == 0)
                    c2 = (((o2 >> sh) == (pref >> sh)) & (((o2 >> b) & 1) == 0)) & first
                c = tl.sum(c0.to(tl.int32) + c1.to(tl.int32) + c2.to(tl.int32))
                take = c <= kk
                kk = tl.where(take, kk - c, kk)
                pref = tl.where(take, pref | (1 << b), pref)
            thr = pref
            kept0 = o0 >= thr
            kept1 = o1 >= thr
            kept2 = o2 >= thr
        else:
            all_true = tl.full([BLOCK], 1, tl.int32) == 1
            kept0 = all_true
            kept1 = all_true
            kept2 = all_true

    if DO_SAMPLE:
        h0 = (seed ^ ((i0 * 0x1E3779B1) & M31) ^ ((step * 0x05EBCA77) & M31)) & M31
        h1 = (seed ^ ((i1 * 0x1E3779B1) & M31) ^ ((step * 0x05EBCA77) & M31)) & M31
        h2 = (seed ^ ((EOS * 0x1E3779B1) & M31) ^ ((step * 0x05EBCA77) & M31)) & M31
        h0 = (h0 ^ (h0 >> 16)) & M31
        h1 = (h1 ^ (h1 >> 16)) & M31
        h2 = (h2 ^ (h2 >> 16)) & M31
        h0 = (h0 * 0x21F0AAAD) & M31
        h1 = (h1 * 0x21F0AAAD) & M31
        h2 = (h2 * 0x21F0AAAD) & M31
        h0 = (h0 ^ (h0 >> 15)) & M31
        h1 = (h1 ^ (h1 >> 15)) & M31
        h2 = (h2 ^ (h2 >> 15)) & M31
        h0 = (h0 * 0x735A2D97) & M31
        h1 = (h1 * 0x735A2D97) & M31
        h2 = (h2 * 0x735A2D97) & M31
        h0 = (h0 ^ (h0 >> 15)) & M31
        h1 = (h1 ^ (h1 >> 15)) & M31
        h2 = (h2 ^ (h2 >> 15)) & M31
        u0 = ((h0 >> 8).to(tl.float32) + 0.5) * 1.1920928955078125e-07
        u1 = ((h1 >> 8).to(tl.float32) + 0.5) * 1.1920928955078125e-07
        u2 = ((h2 >> 8).to(tl.float32) + 0.5) * 1.1920928955078125e-07
        p0 = tl.where(kept0, v0f - tl.log(-tl.log(u0)), float("-inf"))
        p1 = tl.where(kept1, v1f - tl.log(-tl.log(u1)), float("-inf"))
        p2 = tl.where(kept2, v2 - tl.log(-tl.log(u2)), float("-inf"))
    else:

        p0 = v0f
        p1 = v1f
        p2 = v2raw
    bv = tl.maximum(p0, p1)
    bi = tl.where(p1 > p0, i1, i0)
    take2 = first & (p2 > bv)
    bv = tl.where(take2, p2, bv)
    bi = tl.where(take2, EOS, bi)

    m = tl.max(bv, axis=0)
    widx = tl.min(tl.where(bv == m, bi, N + 1), axis=0)
    gtok = tl.load(PhaseRow + widx)
    stop_logit = tl.load(StopIn).to(tl.float32)
    stop_now = (step >= min_tokens) & (stop_logit > 0.0)
    eos_now = widx == EOS
    action = tl.where(stop_now, 1, tl.where(eos_now, 2, 0))


    tl.store(NextTok, gtok)
    tl.store(Hist + emit_pos, gtok)
    tl.store(Action, action)


def _tail_launch(scores, phase_row, stop_buf, next_tok, hist, action,
                 emit_pos, step, min_tokens, temperature, seed,
                 top_k, do_sample, num_warps=32):
    if (scores.numel() != N_LOGITS or scores.dtype != torch.bfloat16
            or not scores.is_cuda or not scores.is_contiguous()):
        raise ValueError("Tail sampler requires contiguous CUDA BF16 phase logits")
    if (phase_row.shape != (N_LOGITS,) or phase_row.dtype != torch.int64
            or not phase_row.is_cuda or not phase_row.is_contiguous()):
        raise ValueError("Tail sampler requires the contiguous CUDA int64 phase row")
    if int(top_k) > 0:
        k_top, do_select = min(int(top_k), N_LOGITS), int(top_k) < N_LOGITS
    else:
        k_top, do_select = N_LOGITS, False
    _tail_kernel[(1,)](
        scores, phase_row, stop_buf, next_tok, hist, action,
        int(emit_pos), int(step), int(min_tokens), float(temperature),
        int(seed) & MASK31, k_top, do_select, bool(do_sample),
        BLOCK, N_LOGITS, EOS_LOCAL, MASK31,
        num_warps=num_warps,
    )
    return next_tok, action
