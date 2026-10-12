"""Fuse indexed sequence gather and multiply, retaining reference reductions.

Sequence attention's broadcast-multiply then sum has a different rounding
order from GEMM. Preserve its PyTorch reductions and chunk accumulation order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _multiply(Q, SEQ, BATCH, O, N: tl.constexpr, BATCHES: tl.constexpr,
              H: tl.constexpr, D: tl.constexpr, START: tl.constexpr,
              COUNT: tl.constexpr, FEATURES: tl.constexpr,
              Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
              BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N * COUNT * H * D
    d = i % D
    h = i // D % H
    s = i // (D * H) % COUNT + START
    n = i // (COUNT * H * D)
    b = tl.load(BATCH + n, valid, 0)
    b = tl.where(b < 0, b + BATCHES, b)
    seq = tl.load(SEQ + b * S0 + s * S1 + h * S2 + d * S3, valid, 0)
    if FEATURES:
        q = tl.load(Q + n * Q0 + s * Q1 + h * Q2, valid, 0)
    else:
        q = tl.load(Q + n * Q0 + h * Q1 + d * Q2, valid, 0)
    tl.store(O + i, q * seq, valid)


def _product(query_or_weights, sequence, batch, start, count, features):
    n = len(query_or_weights)
    batches, _, heads, dim = sequence.shape
    output = torch.empty((n, count, heads, dim), device=sequence.device, dtype=torch.float32)
    with torch.cuda.device(sequence.device):
        _multiply[(triton.cdiv(output.numel(), 512),)](
            query_or_weights, sequence, batch, output, n, batches, heads, dim,
            start, count, features, *query_or_weights.stride(), *sequence.stride(), 512,
        )
    return output


def scores(query, key, batch, scale, chunk_size):
    output = torch.zeros((len(query), key.shape[1], key.shape[2]), device=query.device)
    for start in range(0, key.shape[1], chunk_size):
        count = min(chunk_size, key.shape[1] - start)
        products = _product(query, key, batch, start, count, False)
        output[:, start:start + count] = products.sum(-1) / scale
    return output


def features(weights, value, batch, chunk_size):
    output = torch.zeros((len(weights), value.shape[2], value.shape[3]), device=value.device)
    for start in range(0, value.shape[1], chunk_size):
        count = min(chunk_size, value.shape[1] - start)
        products = _product(weights, value, batch, start, count, True)
        output += products.sum(1)
    return output
