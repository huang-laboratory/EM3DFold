"""FP32 inference attention without materializing [N, K, 2K, H, D].

Keep scaling, bias and softmax in PyTorch to retain its numerical behavior.
Triton gathers projected K/V directly and computes IEEE FP32 dot products.
"""

import math

import torch
from em3dfold.utils.triton_inference import supported_runtime

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _edge_scores_kernel(
        Q, K, I, O,
        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        node = row // C
        neighbor = tl.load(I + row)
        j = tl.arange(0, BLOCK_K)
        d = tl.arange(0, BLOCK_D)
        m = tl.arange(0, 16)
        # First C entries attend to this node, next C to its neighbor.
        source = tl.where(j < C, node, neighbor)
        offsets = ((source * C + j % C) * H + head)[:, None] * D + d[None, :]
        valid = (j[:, None] < 2 * C) & (d[None, :] < D)
        query = tl.load(Q + (row * H + head) * D + d, d < D, 0)
        key = tl.load(K + offsets, valid, 0)
        # tl.dot needs at least 16 rows; only row zero is populated.
        query_matrix = tl.where(m[:, None] == 0, query[None, :], 0.0)
        dot_scores = tl.dot(query_matrix, tl.trans(key), input_precision="ieee")
        scores = tl.sum(dot_scores, axis=0)
        tl.store(O + (row * 2 * C + j) * H + head, scores, j < 2 * C)

    @triton.jit
    def _edge_values_kernel(
        W, V, G, I, O,
        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        node = row // C
        neighbor = tl.load(I + row)
        j = tl.arange(0, BLOCK_K)
        d = tl.arange(0, BLOCK_D)
        m = tl.arange(0, 16)
        source = tl.where(j < C, node, neighbor)
        offsets = ((source * C + j % C) * H + head)[:, None] * D + d[None, :]
        valid = (j[:, None] < 2 * C) & (d[None, :] < D)
        weights = tl.load(W + (row * 2 * C + j) * H + head, j < 2 * C, 0)
        value = tl.load(V + offsets, valid, 0)
        gate = tl.load(G + (row * H + head) * D + d, d < D, 0)
        weights_matrix = tl.where(m[:, None] == 0, weights[None, :], 0.0)
        dot_output = tl.dot(weights_matrix, value, input_precision="ieee")
        output = tl.sum(dot_output, axis=0) * gate
        tl.store(O + (row * H + head) * D + d, output, d < D)


def is_supported(query, key, value, bias, gate, edge_index):
    """Keep unsupported layouts and dtypes on the existing PyTorch path."""
    if triton is None or not supported_runtime() or torch.is_grad_enabled() or query.ndim != 4:
        return False
    n, k, h, d = query.shape
    return (
        n > 0 and h > 0 and 0 < k <= 32 and 0 < d <= 128
        and torch.version.cuda is not None
        and all(t.is_cuda and t.device == query.device and t.dtype == torch.float32
                and t.is_contiguous() for t in (query, key, value, bias, gate))
        and key.shape == query.shape == value.shape == gate.shape
        and bias.shape == (n, k, h)
        and edge_index.shape == (n, k)
        and edge_index.device == query.device
        and edge_index.dtype in (torch.int32, torch.int64)
        and edge_index.is_contiguous()
        and torch.cuda.get_device_capability(query.device)[0] >= 8
    )


def edge_attention(query, key, value, bias, gate, edge_index):
    """Directly gather K/V, retaining the reference FP32 softmax."""
    if not is_supported(query, key, value, bias, gate, edge_index):
        raise RuntimeError("Triton EdgeUpdate requires contiguous NVIDIA CUDA FP32 inference tensors (K <= 32, D <= 128).")
    n, k, h, d = query.shape
    scores = torch.empty((n, k, 2 * k, h), device=query.device, dtype=query.dtype)
    output = torch.empty_like(query)
    block_k = max(16, triton.next_power_of_2(2 * k))
    block_d = max(32, triton.next_power_of_2(d))
    with torch.cuda.device(query.device):
        _edge_scores_kernel[(n * k, h)](
            query, key, edge_index, scores, h, d, k, block_k, block_d,
            num_warps=4,
        )
        self_index = torch.arange(n, device=query.device)[:, None].expand(-1, k)
        attention_bias = torch.cat((bias[self_index], bias[edge_index]), dim=2)
        weights = torch.softmax(scores / math.sqrt(d) + attention_bias, dim=2)
        _edge_values_kernel[(n * k, h)](
            weights, value, gate, edge_index, output, h, d, k, block_k, block_d,
            num_warps=4,
        )
    return output
