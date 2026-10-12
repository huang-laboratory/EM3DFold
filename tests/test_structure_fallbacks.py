import pytest
import torch


@pytest.mark.parametrize("kind", ["node", "outer", "transition", "ipa", "sequence", "window", "frn"])
def test_cpu_autograd_reference_path(kind):
    from em3dfold.models.v3x2.attention import NodeUpdate, OutProductMean, Transition
    from em3dfold.models.v3x2.ipa import InvariantPointAttention
    from em3dfold.models.v3x2.sequence_attention import SequenceAttention
    from em3dfold.scunet.scunet import WMSA
    from em3dfold.scunet.frn import FilterResponseNorm3d
    torch.manual_seed(42)
    node = torch.randn(7, 16, requires_grad=True)
    edge = torch.randn(7, 3, 8, requires_grad=True)
    pos = torch.randn(7, 8)
    indices = torch.randint(7, (7, 3))
    if kind == "node":
        layer, args = NodeUpdate(16, 8, 3, 2, 8), (node, edge, pos, indices)
    elif kind == "outer":
        layer, args = OutProductMean(16, 8, 8), (node, edge, indices)
    elif kind == "transition":
        layer, args = Transition(16, torch.nn.LayerNorm, 2), (node,)
    elif kind == "ipa":
        affines = torch.cat((torch.eye(3).expand(7, 3, 3), torch.randn(7, 3, 1)), -1)
        layer, args = InvariantPointAttention(16, 8, 2, 8, 2, 2), (node, edge, affines, pos, indices)
    elif kind == "sequence":
        layer = SequenceAttention(16, 12, 10, 8, 2, checkpoint=False)
        args = (node, torch.tensor([True, False, True, False, True, False, True]))
    elif kind == "window":
        layer, args = WMSA(16, 16, 16, 3, "SW"), (torch.randn(2, 6, 6, 6, 16, requires_grad=True),)
    else:
        layer, args = FilterResponseNorm3d(8), (torch.randn(2, 8, 3, 3, 3, requires_grad=True),)
    layer.eval()
    layer.inference_backend = "torch"
    expected = layer(*args)
    layer.inference_backend = "triton"
    actual = layer(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert args[0].grad is not None and torch.isfinite(args[0].grad).all()


def test_segmentation_does_not_enable_structure_kernels():
    from em3dfold.scunet.scunet import WMSA
    from em3dfold.scunet.frn import FilterResponseNorm3d
    assert WMSA(16, 16, 16, 3, "W").inference_backend == "torch"
    assert FilterResponseNorm3d(8).inference_backend == "torch"


def test_strict_fp32_inference_argument():
    import argparse
    from em3dfold.infer.inferlm_common import prepare_common_args
    parser = prepare_common_args(argparse.ArgumentParser())
    args = parser.parse_args(["--map", "test.map", "--polymer", "test.cif", "--strict-fp32"])
    assert args.strict_fp32 and not args.fp16


def test_strict_fp32_rejects_half_inference():
    import argparse
    from em3dfold.infer.inferlm_common import run_main
    with pytest.raises(ValueError, match="Strict FP32"):
        run_main(argparse.Namespace(strict_fp32=True, fp16=True), None, {}, None)
