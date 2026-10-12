import importlib.util
from unittest.mock import patch

import pytest
import torch

from em3dfold.models.v3x2.attention import EdgeUpdate


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("n,k,h,d", [(2, 1, 1, 16), (17, 7, 2, 48), (40, 31, 4, 64), (200, 32, 8, 48), (35, 32, 2, 128)])
def test_fp32_inference_matches_reference(n, k, h, d):
    torch.manual_seed(123)
    layer = EdgeUpdate(128, k, h, d, p_drop=0.1).cuda().eval()
    edge = torch.randn(n, k, 128, device="cuda")
    indices = torch.randint(n, (n, k), device="cuda")
    with torch.no_grad():
        layer.attention_backend = "torch"
        reference = layer(edge, indices)
        layer.attention_backend = "triton"
        actual = layer(edge, indices)
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-5)


def test_cpu_inference_and_gradients_use_reference():
    layer = EdgeUpdate(16, 3, 2, 8).eval()
    layer.attention_backend = "auto"
    edge = torch.randn(5, 3, 16, requires_grad=True)
    indices = torch.randint(5, (5, 3))
    with patch.object(layer, "forward_torch", wraps=layer.forward_torch) as reference:
        output = layer(edge, indices)
        output.square().sum().backward()
        assert reference.call_count == 1
    assert edge.grad is not None and torch.isfinite(edge.grad).all()
    with torch.no_grad():
        torch.testing.assert_close(layer(edge, indices), layer.forward_torch(edge, indices))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("training,grad_enabled,dtype", [(True, False, torch.float32), (False, True, torch.float32), (False, False, torch.float16)])
def test_training_autograd_and_half_stay_on_reference(training, grad_enabled, dtype):
    layer = EdgeUpdate(16, 3, 2, 8).cuda().to(dtype).train(training)
    edge = torch.randn(5, 3, 16, device="cuda", dtype=dtype, requires_grad=grad_enabled)
    indices = torch.randint(5, (5, 3), device="cuda")
    with torch.set_grad_enabled(grad_enabled), patch.object(layer, "forward_torch", wraps=layer.forward_torch) as reference:
        output = layer(edge, indices)
        assert reference.call_count == 1
        if grad_enabled:
            output.float().square().sum().backward()
            assert torch.isfinite(edge.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_missing_triton_and_noncontiguous_indices_fall_back():
    from em3dfold.models.v3x2 import triton_edge_attention
    layer = EdgeUpdate(16, 3, 2, 8).cuda().eval()
    edge = torch.randn(5, 3, 16, device="cuda")
    indices = torch.randint(5, (3, 5), device="cuda").t()
    with torch.no_grad(), patch.object(layer, "forward_torch", wraps=layer.forward_torch) as reference:
        layer(edge, indices)
        assert reference.call_count == 1
    with torch.no_grad(), patch.object(triton_edge_attention, "triton", None):
        torch.testing.assert_close(layer(edge, indices.contiguous()), layer.forward_torch(edge, indices.contiguous()))
