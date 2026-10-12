"""Compare complete FP32 structure-network forwards with PyTorch and Triton.

Run with the repository's src directory on PYTHONPATH. The input file contains
the CPU keyword arguments accepted by Model.forward, saved using torch.save.
Only load checkpoints and input files from a trusted source.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from em3dfold.models.v3x2.attention import EdgeUpdate
from em3dfold.models.v3x2.model import Model
from em3dfold.utils.conv_precision import InferenceConv3d


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def output_tensors(output):
    tensors = {}
    for key, value in output.items():
        if torch.is_tensor(value):
            tensors[key] = value.detach().cpu()
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if torch.is_tensor(item):
                    tensors[f"{key}.{index}"] = item.detach().cpu()
    return tensors


def compare(reference, candidate):
    results = {}
    for key, ref in reference.items():
        actual = candidate[key]
        if not ref.is_floating_point():
            results[key] = {"equal": torch.equal(ref, actual)}
            continue
        difference = (ref - actual).double()
        item = {
            "max_abs": difference.abs().max().item(),
            "rms": difference.square().mean().sqrt().item(),
            "finite": bool(torch.isfinite(actual).all()),
            "reference_max_abs": ref.abs().max().item(),
            "within_mixed_tolerance": torch.allclose(ref, actual, atol=1e-3, rtol=1e-4),
        }
        if key.startswith("pred_positions."):
            item["max_position_distance_A"] = difference.norm(dim=-1).max().item()
        results[key] = item
    return results


def compare_predictions(reference, candidate):
    results = {}
    for key in ("pred_aa_logits", "pred_edge_aa_logits", "pred_edge_existence", "pred_ss_logits"):
        if key not in reference:
            continue
        ref, actual = reference[key], candidate[key]
        if key == "pred_edge_aa_logits":
            ref, actual = ref.flatten(-2), actual.flatten(-2)
        if key == "pred_aa_logits":
            # Match the model's separate protein and nucleotide distributions.
            mask = reference["prot_mask"].bool()
            groups = [("protein", ref[mask, :20], actual[mask, :20]),
                      ("na", ref[~mask, 20:], actual[~mask, 20:])]
        else:
            groups = [(key, ref, actual)]
        for label, ref_group, actual_group in groups:
            if ref_group.numel() == 0:
                continue
            ref_probs, actual_probs = ref_group.softmax(-1), actual_group.softmax(-1)
            results[label] = {
                "max_probability_error": (ref_probs - actual_probs).abs().max().item(),
                "changed_argmax": int((ref_probs.argmax(-1) != actual_probs.argmax(-1)).sum()),
                "predictions": ref_probs.numel() // ref_probs.shape[-1],
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tf32", action="store_true", help="Match CLI TF32 policy; disabled by default for strict FP32")
    parser.add_argument("--ops", default="all", help="all or comma-separated edge,rope,outer,swiglu,frn,window,sequence,rbf")
    parser.add_argument("--include-edge-baseline", action="store_true", help="Also time the previous EdgeUpdate-only acceleration")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_args = yaml.safe_load(Path(args.config).read_text())["model"]["args"]
    model = Model(**model_args).eval()
    # This benchmark compares FP32 backends even if the caller's environment
    # enables the separate conv-only mixed-precision optimization.
    for layer in model.modules():
        if isinstance(layer, InferenceConv3d):
            layer.conv_fp16 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model = model.float().to(args.device)
    inputs = to_device(torch.load(args.input, map_location="cpu", weights_only=True), args.device)
    inputs["run_iters"] = args.run_iters
    layers = [layer for layer in model.modules() if isinstance(layer, EdgeUpdate)]
    accelerated = [layer for layer in model.modules() if hasattr(layer, "inference_backend")]
    selected = set(args.ops.split(","))
    valid = {"all", "edge", "rope", "outer", "swiglu", "frn", "window", "sequence", "rbf"}
    if not selected <= valid:
        raise ValueError(f"Unknown ops: {selected - valid}")
    measurements = {backend: [] for backend in ("torch", "triton")}
    if args.include_edge_baseline:
        measurements = {backend: [] for backend in ("torch", "edge", "triton")}
    peaks = {backend: [] for backend in measurements}
    outputs = {}

    def run(backend):
        for layer in layers:
            layer.attention_backend = "triton" if backend == "edge" else (backend if "all" in selected or "edge" in selected else "torch")
        for layer in accelerated:
            layer.inference_backend = backend if backend != "edge" and ("all" in selected or layer.triton_op_name in selected) else "torch"
        with torch.no_grad():
            return model(**inputs)

    # Warm both paths, including Triton compilation, outside timing.
    for backend in measurements:
        print(f"Warming {backend}", flush=True)
        result = run(backend)
        torch.cuda.synchronize(args.device)
        outputs[backend] = output_tensors(result)
        del result
    baseline_repeat = output_tensors(run("torch"))
    for repeat in range(args.repeats):
        # Alternate order to reduce clock/temperature bias.
        order = tuple(measurements) if repeat % 2 == 0 else tuple(reversed(measurements))
        for backend in order:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter()
            result = run(backend)
            torch.cuda.synchronize(args.device)
            duration = time.perf_counter() - start
            measurements[backend].append(duration)
            peaks[backend].append(torch.cuda.max_memory_allocated(args.device) / 2**20)
            del result
            print(f"{repeat + 1}/{args.repeats} {backend}: {duration:.4f}s", flush=True)
    medians = {backend: statistics.median(values) for backend, values in measurements.items()}
    report = {
        "scope": "complete pretrained v3x2 forward; excludes preprocessing, language-model embedding and structure postprocessing",
        "checkpoint": args.checkpoint, "config": args.config, "input": args.input,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
        "dtype": "float32", "tf32": args.tf32, "conv_fp16": False, "nodes": len(inputs["affines"]),
        "triton_ops": args.ops,
        "run_iters": args.run_iters, "edge_update_calls_per_forward": len(layers) * args.run_iters,
        "seconds": measurements, "median_seconds": medians,
        "speedup": medians["torch"] / medians["triton"],
        "peak_allocated_MiB": {backend: max(values) for backend, values in peaks.items()},
        "output_errors": compare(outputs["torch"], outputs["triton"]),
        "reference_repeat_errors": compare(outputs["torch"], baseline_repeat),
        "prediction_errors": compare_predictions(outputs["torch"], outputs["triton"]),
        "accuracy_criteria": "all float outputs atol=1e-3, rtol=1e-4; positions distance <=1e-3 A; probability error <=1e-3; argmax unchanged; integer outputs equal",
    }
    report["accuracy_pass"] = all(
        item.get("equal", True) and item.get("finite", True)
        and item.get("within_mixed_tolerance", True)
        and item.get("max_position_distance_A", 0) <= 1e-3
        for item in report["output_errors"].values()
    ) and all(item["max_probability_error"] <= 1e-3 and item["changed_argmax"] == 0
              for item in report["prediction_errors"].values())
    if "edge" in medians:
        report["speedup_over_edge"] = medians["edge"] / medians["triton"]
        report["edge_output_errors"] = compare(outputs["torch"], outputs["edge"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("median_seconds", "speedup", "peak_allocated_MiB", "accuracy_pass")}, indent=2))
    if not report["accuracy_pass"]:
        raise SystemExit("Full-model output comparison failed accuracy criteria; see report")


if __name__ == "__main__":
    main()
