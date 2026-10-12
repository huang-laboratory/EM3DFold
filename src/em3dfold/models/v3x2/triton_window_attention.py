"""Direct QKV window dot products; retain PyTorch bias, mask and softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def _layout(X, O, TOTAL: tl.constexpr, Z: tl.constexpr, Y: tl.constexpr,
            NX: tl.constexpr, C: tl.constexpr, P: tl.constexpr, SHIFT: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
            S4: tl.constexpr, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = i % C
    nz, ny, nx = Z // P, Y // P, NX // P
    if INVERSE:
        batch = i // (Z * Y * NX * C)
        z = (i // (Y * NX * C) % Z - SHIFT + Z) % Z
        y = (i // (NX * C) % Y - SHIFT + Y) % Y
        x = (i // C % NX - SHIFT + NX) % NX
        window = (z // P * ny + y // P) * nx + x // P
        token = (z % P * P + y % P) * P + x % P
        offset = batch * S0 + window * S1 + token * S2 + channel * S3
    else:
        batch = i // (Z * Y * NX * C)
        window = i // (P * P * P * C) % (nz * ny * nx)
        token = i // C % (P * P * P)
        z = (window // (ny * nx) * P + token // (P * P) + SHIFT) % Z
        y = (window // nx % ny * P + token // P % P + SHIFT) % Y
        x = (window % nx * P + token % P + SHIFT) % NX
        offset = batch * S0 + z * S1 + y * S2 + x * S3 + channel * S4
    value = tl.load(X + offset, i < TOTAL, 0)
    tl.store(O + i, value, i < TOTAL)


def partition(x, window_size, shift):
    b, z, y, width, c = x.shape
    output = torch.empty((b, (z // window_size) * (y // window_size) * (width // window_size),
                          window_size ** 3, c), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _layout[(triton.cdiv(x.numel(), 512),)](
            x, output, x.numel(), z, y, width, c, window_size, shift,
            *x.stride(), False, 512,
        )
    return output


def unpartition(x, spatial_shape, window_size, shift):
    z, y, width = spatial_shape
    b, _, _, c = x.shape
    output = torch.empty((b, z, y, width, c), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _layout[(triton.cdiv(x.numel(), 512),)](
            x, output, x.numel(), z, y, width, c, window_size, shift,
            *x.stride(), 0, True, 512,
        )
    return output


@triton.jit
def _scores(QKV, O, WINDOWS: tl.constexpr, P: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr):
    window = tl.program_id(0)
    head = tl.program_id(1)
    p = tl.arange(0, BP)
    d = tl.arange(0, BD)
    offsets = window * P * 3 * H * D + p[:, None] * 3 * H * D + head * D + d[None, :]
    valid = (p[:, None] < P) & (d[None, :] < D)
    q = tl.load(QKV + offsets, valid, 0)
    k = tl.load(QKV + offsets + H * D, valid, 0)
    scores = tl.dot(q, tl.trans(k), input_precision="ieee")
    out_offsets = (head * WINDOWS + window) * P * P + p[:, None] * P + p[None, :]
    tl.store(O + out_offsets, scores, (p[:, None] < P) & (p[None, :] < P))


@triton.jit
def _values(QKV, W, O, WINDOWS: tl.constexpr, P: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr):
    window = tl.program_id(0)
    head = tl.program_id(1)
    p = tl.arange(0, BP)
    d = tl.arange(0, BD)
    weights = tl.load(W + (head * WINDOWS + window) * P * P + p[:, None] * P + p[None, :],
                      (p[:, None] < P) & (p[None, :] < P), 0)
    offsets = window * P * 3 * H * D + p[:, None] * 3 * H * D + (2 * H + head) * D + d[None, :]
    value = tl.load(QKV + offsets, (p[:, None] < P) & (d[None, :] < D), 0)
    output = tl.dot(weights, value, input_precision="ieee")
    out_offsets = window * P * H * D + p[:, None] * H * D + head * D + d[None, :]
    tl.store(O + out_offsets, output, (p[:, None] < P) & (d[None, :] < D))


def supported(qkv, heads, head_dim):
    return (qkv.ndim == 4 and qkv.is_contiguous() and 0 < qkv.shape[2] <= 64
            and heads > 0 and 0 < head_dim <= 128
            and qkv.shape[-1] == 3 * heads * head_dim)


def window_scores(qkv, heads, head_dim):
    b, w, p, _ = qkv.shape
    output = torch.empty((heads, b, w, p, p), device=qkv.device, dtype=qkv.dtype)
    with torch.cuda.device(qkv.device):
        _scores[(b * w, heads)](qkv, output, b * w, p, heads, head_dim,
                               max(16, triton.next_power_of_2(p)),
                               max(32, triton.next_power_of_2(head_dim)), num_warps=4)
    return output


def window_values(qkv, weights, heads, head_dim):
    b, w, p, _ = qkv.shape
    output = torch.empty((b, w, p, heads * head_dim), device=qkv.device, dtype=qkv.dtype)
    with torch.cuda.device(qkv.device):
        _values[(b * w, heads)](qkv, weights, output, b * w, p, heads, head_dim,
                               max(32, triton.next_power_of_2(p)),
                               max(16, triton.next_power_of_2(head_dim)), num_warps=4)
    return output
