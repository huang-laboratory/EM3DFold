"""Inference kernels with separate FP32 rounding for each reference operation."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rope(Q, K, P, I, OQ, OK, N: tl.constexpr, C: tl.constexpr,
          H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    neighbor = tl.load(I + i // (H * D), i < N * C * H * D, 0)
    neighbor = tl.where(neighbor < 0, neighbor + N, neighbor)
    pair = d // 2 * 2
    cos = tl.load(P + neighbor * D + pair + 1, i < N * C * H * D, 0)
    sin = tl.load(P + neighbor * D + pair, i < N * C * H * D, 0)
    k = tl.load(K + i, i < N * C * H * D, 0)
    kr = tl.load(K + i - d + tl.where(d % 2 == 0, d + 1, d - 1), i < N * C * H * D, 0)
    kr = tl.where(d % 2 == 0, -kr, kr)
    tl.store(OK + i, k * cos + kr * sin, i < N * C * H * D)
    # Q is smaller; only the first N*H*D lanes write it.
    qnode = i // (H * D)
    q = tl.load(Q + i, i < N * H * D, 0)
    qr = tl.load(Q + i - d + tl.where(d % 2 == 0, d + 1, d - 1), i < N * H * D, 0)
    qr = tl.where(d % 2 == 0, -qr, qr)
    qc = tl.load(P + qnode * D + pair + 1, i < N * H * D, 0)
    qs = tl.load(P + qnode * D + pair, i < N * H * D, 0)
    tl.store(OQ + i, q * qc + qr * qs, i < N * H * D)


def rope(query, key, positions, indices):
    n, c, h, d = key.shape
    if not (n > 0 and c > 0 and d % 2 == 0 and query.shape == (n, h, d)
            and query.is_contiguous() and key.is_contiguous()
            and positions.is_contiguous() and positions.shape == (n, d)
            and indices.is_contiguous() and indices.device == query.device
            and positions.device == query.device and indices.dtype in (torch.int32, torch.int64)):
        return None
    oq, ok = torch.empty_like(query), torch.empty_like(key)
    with torch.cuda.device(query.device):
        _rope[(triton.cdiv(key.numel(), 512),)](
            query, key, positions, indices, oq, ok, n, c, h, d, 512,
            enable_fp_fusion=False,
        )
    return oq, ok


@triton.jit
def _outer(L, R, I, O, TOTAL: tl.constexpr, N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
           LS0: tl.constexpr, LS1: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
           BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // (C * C)
    n = row // K
    neighbor = tl.load(I + row, i < TOTAL, 0)
    neighbor = tl.where(neighbor < 0, neighbor + N, neighbor)
    l = tl.load(L + n * LS0 + (i // C % C) * LS1, i < TOTAL, 0)
    r = tl.load(R + neighbor * RS0 + (i % C) * RS1, i < TOTAL, 0)
    tl.store(O + i, l * r, i < TOTAL)


def outer_product(left, right, indices):
    n, c = left.shape
    k = indices.shape[1]
    output = torch.empty((n, k, c * c), device=left.device, dtype=left.dtype)
    with torch.cuda.device(left.device):
        _outer[(triton.cdiv(output.numel(), 512),)](
            left, right, indices, output, output.numel(), n, c, k,
            *left.stride(), *right.stride(), 512,
        )
    return output


@triton.jit
def _swiglu(A, B, O, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < TOTAL, 0)
    b = tl.load(B + i, i < TOTAL, 0)
    silu = tl.div_rn(a, 1.0 + libdevice.exp(-a))
    tl.store(O + i, silu * b, i < TOTAL)


def swiglu(a, b):
    output = torch.empty_like(a)
    with torch.cuda.device(a.device):
        _swiglu[(triton.cdiv(a.numel(), 512),)](
            a, b, output, a.numel(), 512, enable_fp_fusion=False,
        )
    return output


@triton.jit
def _frn(X, SCALE, GAMMA, BETA, TAU, O, TOTAL: tl.constexpr,
         C: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    nc = i // SPATIAL
    channel = nc % C
    x = tl.load(X + i, i < TOTAL, 0)
    scale = tl.load(SCALE + nc, i < TOTAL, 0)
    gamma = tl.load(GAMMA + channel, i < TOTAL, 0)
    beta = tl.load(BETA + channel, i < TOTAL, 0)
    tau = tl.load(TAU + channel, i < TOTAL, 0)
    y = gamma * (x * scale) + beta
    output = tl.where((y != y) | (tau != tau), float("nan"), tl.maximum(y, tau))
    tl.store(O + i, output, i < TOTAL)


def frn_affine(x, scale, gamma, beta, tau):
    output = torch.empty_like(x)
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    with torch.cuda.device(x.device):
        _frn[(triton.cdiv(x.numel(), 512),)](
            x, scale, gamma, beta, tau, output, x.numel(), x.shape[1], spatial,
            512, enable_fp_fusion=False,
        )
    return output


@triton.jit
def _rbf(DIST, MU, O, TOTAL: tl.constexpr, K: tl.constexpr, F: tl.constexpr,
         BINS: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
         INV_SIGMA: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = i // BINS
    offset = source // (K * F) * S0 + source // F % K * S1 + source % F * S2
    distance = tl.load(DIST + offset, i < TOTAL, 0)
    mu = tl.load(MU + i % BINS, i < TOTAL, 0)
    delta = (distance - mu) * INV_SIGMA
    result = libdevice.exp(-(delta * delta))
    tl.store(O + i, result, i < TOTAL)


def radial_basis(distances, bins=64, lower=0.5, upper=30.5):
    n, k, features = distances.shape
    mu = torch.linspace(lower, upper, bins, device=distances.device)
    output = torch.empty((n, k, features, bins), device=distances.device, dtype=torch.float32)
    # PyTorch's CPU-scalar division uses multiplication by the FP32 reciprocal.
    inverse_sigma = float(torch.tensor((upper - lower) / bins, dtype=torch.float32).reciprocal())
    with torch.cuda.device(distances.device):
        _rbf[(triton.cdiv(output.numel(), 512),)](
            distances, mu, output, output.numel(), k, features, bins,
            *distances.stride(), inverse_sigma, 512, enable_fp_fusion=False,
        )
    return output
