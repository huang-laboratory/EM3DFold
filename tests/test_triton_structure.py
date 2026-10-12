import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton required",
)


@pytest.mark.parametrize("n,k,h,d", [(17, 7, 2, 48), (200, 32, 8, 48), (3, 2, 1, 16)])
def test_rope(n, k, h, d):
    from em3dfold.models.modules import qk_modulation
    from em3dfold.models.v3x2.triton_pointwise import rope
    q = torch.randn(n, h, d, device="cuda")
    key = torch.randn(n, k, h, d, device="cuda")
    positions = torch.randn(n, d, device="cuda")
    indices = torch.randint(n, (n, k), device="cuda")
    with torch.no_grad():
        expected = qk_modulation(q, key, positions, indices)
        actual = rope(q, key, positions, indices)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_outer_product_strided_chunks():
    from em3dfold.models.v3x2.triton_pointwise import outer_product
    left, right = torch.randn(17, 96, device="cuda").chunk(2, -1)
    indices = torch.randint(17, (17, 7), device="cuda")
    expected = torch.einsum("ni,nkj->nkij", left, right[indices]).flatten(-2)
    with torch.no_grad():
        actual = outer_product(left, right, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_swiglu():
    from em3dfold.models.v3x2.triton_pointwise import swiglu
    a, b = torch.randn(6400, 256, device="cuda"), torch.randn(6400, 256, device="cuda")
    with torch.no_grad():
        actual = swiglu(a, b)
        expected = torch.nn.functional.silu(a) * b
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("shape", [(3, 7, 6, 6, 6), (3, 64, 12, 12, 12)])
def test_frn(shape):
    from em3dfold.scunet.frn import FilterResponseNorm3d
    layer = FilterResponseNorm3d(shape[1]).cuda().eval()
    torch.nn.init.normal_(layer.gamma)
    torch.nn.init.normal_(layer.beta)
    torch.nn.init.normal_(layer.tau)
    x = torch.randn(shape, device="cuda")
    with torch.no_grad():
        layer.inference_backend = "torch"
        expected = layer(x)
        layer.inference_backend = "triton"
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("heads,kind", [(1, "W"), (2, "W"), (1, "SW"), (2, "SW")])
def test_window_attention(heads, kind):
    from em3dfold.scunet.scunet import WMSA
    layer = WMSA(heads * 16, heads * 16, 16, 3, kind).cuda().eval()
    x = torch.randn(2, 6, 6, 6, heads * 16, device="cuda")
    with torch.no_grad():
        layer.inference_backend = "torch"
        expected = layer(x)
        layer.inference_backend = "triton"
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("length,chunk", [(17, 7), (401, 200), (33, 200)])
def test_sequence_attention_reduction_order(length, chunk):
    from em3dfold.models.v3x2.sequence_attention import get_batched_sequence_attention_scores, get_batched_sequence_attention_features
    from em3dfold.models.v3x2.triton_sequence_attention import scores, features
    q = torch.randn(17, 2, 48, device="cuda")
    k = torch.randn(3, length, 2, 48, device="cuda")
    v = torch.randn_like(k)
    batch = torch.randint(3, (17,), device="cuda")
    weights = torch.randn(17, length, 2, device="cuda").softmax(1)
    with torch.no_grad():
        expected = get_batched_sequence_attention_scores(q, k, batch, 48**0.5, chunk, "cuda")
        actual = scores(q, k, batch, 48**0.5, chunk)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected = get_batched_sequence_attention_features(weights, v, batch, chunk, "cuda")
        actual = features(weights, v, batch, chunk)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("broadcast", [False, True])
def test_radial_basis(broadcast):
    from em3dfold.models.v3x2.track import rbf
    from em3dfold.models.v3x2.triton_pointwise import radial_basis
    distances = torch.rand(17, 7, 1 if broadcast else 25, device="cuda") * 40
    if broadcast:
        distances = distances.expand(-1, -1, 25)
    with torch.no_grad():
        expected = rbf(distances, d_count=32)
        actual = radial_basis(distances, bins=32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_half_autocast_uses_reference():
    from unittest.mock import patch
    from em3dfold.models.v3x2.attention import Transition
    layer = Transition(16, torch.nn.LayerNorm, 2).cuda().eval()
    layer.inference_backend = "triton"
    x = torch.randn(7, 16, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        with patch("em3dfold.models.v3x2.triton_pointwise.swiglu", side_effect=AssertionError("FP16 must use reference")):
            actual = layer(x)
        layer.inference_backend = "torch"
        expected = layer(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_sequence_module_mixed_nodes_and_padding():
    from em3dfold.models.v3x2.sequence_attention import SequenceAttention
    layer = SequenceAttention(16, 12, 10, 8, 2, checkpoint=False).cuda().eval()
    node = torch.randn(17, 16, device="cuda")
    mask = torch.arange(17, device="cuda") % 2 == 0
    batch = torch.arange(17, device="cuda") % 3
    kwargs = dict(prot_seq_emb=torch.randn(3, 401, 12, device="cuda"),
                  na_seq_emb=torch.randn(3, 21, 10, device="cuda"),
                  prot_seq_mask=torch.rand(3, 401, device="cuda") > 0.25,
                  na_seq_mask=torch.rand(3, 21, device="cuda") > 0.25)
    kwargs["prot_seq_mask"][0] = False
    with torch.no_grad():
        layer.inference_backend = "torch"
        expected = layer(node, mask, batch, **kwargs)
        layer.inference_backend = "triton"
        actual = layer(node, mask, batch, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_tf32_keeps_reference_attention_products():
    from unittest.mock import patch
    from em3dfold.models.v3x2.attention import EdgeUpdate
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        layer = EdgeUpdate(16, 3, 2, 8).cuda().eval()
        layer.attention_backend = "triton"
        x = torch.randn(7, 3, 16, device="cuda")
        indices = torch.randint(7, (7, 3), device="cuda")
        with torch.no_grad(), patch.object(layer, "forward_torch", wraps=layer.forward_torch) as reference:
            actual = layer(x, indices)
            assert reference.call_count == 1
            torch.testing.assert_close(actual, layer.forward_torch(x, indices), atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_older_or_missing_runtime_uses_reference():
    from unittest.mock import patch
    from em3dfold.models.v3x2.attention import EdgeUpdate
    layer = EdgeUpdate(16, 3, 2, 8).cuda().eval()
    layer.attention_backend = "auto"
    x = torch.randn(7, 3, 16, device="cuda")
    indices = torch.randint(7, (7, 3), device="cuda")
    with torch.no_grad(), patch("em3dfold.models.v3x2.attention.supported_runtime", return_value=False):
        with patch.object(layer, "forward_torch", wraps=layer.forward_torch) as reference:
            actual = layer(x, indices)
            assert reference.call_count == 1
        torch.testing.assert_close(actual, layer.forward_torch(x, indices), atol=0, rtol=0)
