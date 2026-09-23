import triton
import triton.language as tl

@triton.jit
def _partials(Q, NewK, NewV, Cos, Sin, K, V, Position, Partial, Lse,
              CAP: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr,
              ROTARY: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    d = tl.arange(0, 128)
    position = tl.load(Position)
    q = tl.load(Q + head * 128 + d).to(tl.float32)
    nk = tl.load(NewK + (head // 4) * 128 + d).to(tl.float32)
    nv = tl.load(NewV + (head // 4) * 128 + d).to(tl.float32)
    if ROTARY:
        cosine = tl.load(Cos + d).to(tl.float32)
        sine = tl.load(Sin + d).to(tl.float32)
        qp = tl.load(Q + head * 128 + (d ^ 1)).to(tl.float32)
        kp = tl.load(NewK + (head // 4) * 128 + (d ^ 1)).to(tl.float32)
        qp = tl.where(d % 2 == 0, -qp, qp)
        kp = tl.where(d % 2 == 0, -kp, kp)
        q = (q * cosine + qp * sine).to(tl.bfloat16).to(tl.float32)
        nk = (nk * cosine + kp * sine).to(tl.bfloat16).to(tl.float32)


    if (head % 4 == 0) & (split == 0):
        offset = (head // 4) * CAP * 128 + position * 128 + d
        tl.store(K + offset, nk, (position >= 0) & (position < CAP))
        tl.store(V + offset, nv, (position >= 0) & (position < CAP))
    output = tl.full((128,), 0, tl.float32)
    lse = tl.full((), -float("inf"), tl.float32)
    start = split * BLOCK
    if start <= position:
        t = start + tl.arange(0, BLOCK)
        valid = (t <= position) & (t < CAP)
        old = valid & (t != position)
        offset = (head // 4) * CAP * 128 + t[:, None] * 128 + d[None, :]
        key = tl.load(K + offset, old[:, None], other=0).to(tl.float32)
        value = tl.load(V + offset, old[:, None], other=0).to(tl.float32)
        key = tl.where(t[:, None] == position, nk[None, :], key)
        value = tl.where(t[:, None] == position, nv[None, :], value)
        scores = tl.sum(key * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - maximum)
        denominator = tl.sum(probabilities, axis=0)
        output = tl.sum(value * probabilities[:, None], axis=0) / denominator
        lse = maximum + tl.log(denominator)
    tl.store(Partial + (head * SPLITS + split) * 128 + d, output)
    tl.store(Lse + head * SPLITS + split, lse)
