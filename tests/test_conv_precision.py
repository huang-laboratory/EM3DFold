import argparse
import pickle

import pytest
import torch
from torch import nn

from em3dfold.utils.conv_precision import InferenceConv3d, enable_structure_conv_fp16


def test_checkpoint_cpu_autograd_and_pickle():
    model = nn.Sequential(nn.Conv3d(2, 4, 3, padding=1), nn.ReLU())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    weight = model[0].weight
    x = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
    expected = model(x)
    enable_structure_conv_fp16(model)
    assert model[0].weight is weight
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
    model(x).sum().backward()
    assert x.grad is not None
    restored = pickle.loads(pickle.dumps(model))
    torch.testing.assert_close(restored(x), expected, atol=0, rtol=0)


def test_default_and_structure_only(monkeypatch):
    from em3dfold.models.v3x2.cryo_init import CryoInit
    monkeypatch.delenv('EM3DFOLD_CONV_FP16', raising=False)
    assert not any(isinstance(m, InferenceConv3d) for m in CryoInit().modules())
    segmentation = nn.Sequential(nn.Conv3d(1, 2, 3))
    monkeypatch.setenv('EM3DFOLD_CONV_FP16', '1')
    model = CryoInit()
    assert isinstance(model.conv_rectangle_pre, InferenceConv3d)
    assert all(isinstance(m, InferenceConv3d) for m in model.modules() if isinstance(m, nn.Conv3d))
    assert not isinstance(segmentation[0], InferenceConv3d)


def test_cli_policy_and_environment_restored(monkeypatch):
    from em3dfold.infer import inferlm_common as common
    monkeypatch.delenv('EM3DFOLD_CONV_FP16', raising=False)
    parser = common.prepare_common_args(argparse.ArgumentParser())
    args = parser.parse_args(['--map', 'test.map', '--polymer', 'test.cif', '--conv-fp16', '--strict-fp32'])
    def fake_run(*unused):
        import os
        assert os.environ['EM3DFOLD_CONV_FP16'] == '1'
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        raise RuntimeError('test cleanup')
    monkeypatch.setattr(common, '_run_main', fake_run)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        with pytest.raises(RuntimeError, match='test cleanup'):
            common.run_main(args, None, {}, None)
        import os
        assert 'EM3DFOLD_CONV_FP16' not in os.environ
        args.strict_fp32 = False
        args.fp16 = True
        with pytest.raises(ValueError, match='Conv-only'):
            common.run_main(args, None, {}, None)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_matches_local_autocast_and_fallbacks():
    model = nn.Sequential(nn.Conv3d(2, 4, 3, padding=1)).cuda().eval()
    x = torch.randn(2, 2, 5, 5, 5, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        expected = model(x).float()
    enable_structure_conv_fp16(model)
    with torch.no_grad():
        actual = model(x)
    assert actual.dtype == torch.float32 and model[0].weight.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        model[0].conv_fp16 = False
        native = model(x)
        model[0].conv_fp16 = True
        model.train()
        torch.testing.assert_close(model(x), native, atol=0, rtol=0)
    model.eval()
    x.requires_grad_()
    actual = model(x)
    reference = nn.Conv3d.forward(model[0], x)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    actual.sum().backward()
    assert x.grad is not None
