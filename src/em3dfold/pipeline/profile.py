import argparse
import importlib
import time
from collections import defaultdict

import torch
from omegaconf import OmegaConf


def add_args(parser):
    parser.add_argument(
        "--config",
        default="em3dfold/infer/config/model_v3x2_12l_256_128_h8.yaml",
        help="Path to model config yaml",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--n-res", type=int, default=128, help="Number of residues")
    parser.add_argument("--prot-seq-len", type=int, default=512, help="Protein sequence length")
    parser.add_argument("--na-seq-len", type=int, default=256, help="Nucleic acid sequence length")
    parser.add_argument("--map-size", type=int, default=64, help="Cubic cryo grid size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forward steps")
    parser.add_argument("--steps", type=int, default=3, help="Measured forward steps")
    parser.add_argument("--run-iters", type=int, default=1, help="Model recycle/run_iters")
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of timing entries to print",
    )
    return parser


class TimerStore:
    def __init__(self, device: str):
        self.device = device
        self.stats = defaultdict(lambda: {"time": 0.0, "calls": 0})

    def sync(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)

    def wrap_callable(self, name, fn):
        def wrapped(*args, **kwargs):
            self.sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.sync()
            dt = time.perf_counter() - t0
            self.stats[name]["time"] += dt
            self.stats[name]["calls"] += 1
            return out

        return wrapped


def _load_model(config_path, device):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    module = importlib.import_module(model_cfg.module)
    model_cls = getattr(module, model_cfg.class_name)
    model = model_cls(**OmegaConf.to_container(model_cfg.args, resolve=True))
    model.to(device)
    model.eval()
    return model, OmegaConf.to_container(model_cfg.args, resolve=True)


def _make_affines(n_res, device):
    affines = torch.zeros(n_res, 3, 4, device=device)
    affines[:, :, :3] = torch.eye(3, device=device).unsqueeze(0)
    return affines


def _make_inputs(model_args, n_res, prot_seq_len, na_seq_len, map_size, device):
    d_seq = model_args.get("d_seq", 1280)
    d_seq_na = model_args.get("d_seq_na", d_seq)

    prot_mask = torch.zeros(n_res, device=device, dtype=torch.long)
    prot_mask[: n_res // 2] = 1
    batch = torch.zeros(n_res, device=device, dtype=torch.long)

    return {
        "affines": _make_affines(n_res, device),
        "prot_mask": prot_mask,
        "cryo_grids": [torch.zeros(1, 1, map_size, map_size, map_size, device=device)],
        "cryo_global_origins": [torch.zeros(3, device=device)],
        "cryo_voxel_sizes": [torch.ones(3, device=device)],
        "prot_seq_embed": torch.randn(1, prot_seq_len, d_seq, device=device),
        "prot_seq_embed_mask": torch.ones(1, prot_seq_len, dtype=torch.bool, device=device),
        "na_seq_embed": torch.randn(1, na_seq_len, d_seq_na, device=device),
        "na_seq_embed_mask": torch.ones(1, na_seq_len, dtype=torch.bool, device=device),
        "batch": batch,
    }


def _attach_timers(model, timers):
    originals = []

    def wrap_module_forward(name, module):
        if module is None:
            return
        original_forward = module.forward
        originals.append((module, original_forward))
        module.forward = timers.wrap_callable(name, original_forward)

    wrap_module_forward("model", model)
    wrap_module_forward("init", model.init)
    wrap_module_forward("embed_mol_type", model.embed_mol_type)
    wrap_module_forward("aa_predictor", model.aa_predictor)
    wrap_module_forward("rmsd_predictor", model.rmsd_predictor)
    if getattr(model, "pred_edge_exist", False):
        wrap_module_forward("edge_existence_predictor", model.edge_existence_predictor)
    if getattr(model, "pred_node_exist", False):
        wrap_module_forward("node_existence_predictor", model.node_existence_predictor)
    if getattr(model, "pred_pairing", False):
        wrap_module_forward("pairing_predictor", model.pairing_predictor)

    if getattr(model, "use_seq_attn", False):
        for i, module in enumerate(model.seq_attn_blocks):
            wrap_module_forward(f"seq_attn[{i}]", module)

    for i, block in enumerate(model.blocks):
        wrap_module_forward(f"block[{i}]", block)
        wrap_module_forward(f"block[{i}].node_update", block.node_update)
        wrap_module_forward(f"block[{i}].node_transition", block.node_transition)
        wrap_module_forward(f"block[{i}].out_product", block.out_product)
        wrap_module_forward(f"block[{i}].edge_update", block.edge_update)
        wrap_module_forward(f"block[{i}].edge_transition", block.edge_transition)
        wrap_module_forward(f"block[{i}].ipa", block.ipa)
        wrap_module_forward(f"block[{i}].ipa_transition", block.ipa_transition)
        if block.enable_geometry_update:
            wrap_module_forward(f"block[{i}].bb_update", block.bb_update)
            wrap_module_forward(f"block[{i}].torsion_update", block.torsion_update)

    model._run_cryo_init = timers.wrap_callable("model._run_cryo_init", model._run_cryo_init)
    model._compute_edge_bias = timers.wrap_callable(
        "model._compute_edge_bias", model._compute_edge_bias
    )
    return originals


def _print_results(timers, steps, topk):
    rows = []
    for name, stat in timers.stats.items():
        total_ms = stat["time"] * 1000.0
        rows.append(
            (
                name,
                total_ms,
                total_ms / max(stat["calls"], 1),
                stat["calls"],
                total_ms / max(steps, 1),
            )
        )

    rows.sort(key=lambda x: x[1], reverse=True)
    print("name\ttotal_ms\tavg_call_ms\tcalls\tavg_step_ms")
    for name, total_ms, avg_call_ms, calls, avg_step_ms in rows[:topk]:
        print(
            f"{name}\t{total_ms:.3f}\t{avg_call_ms:.3f}\t{calls}\t{avg_step_ms:.3f}"
        )


def main(args):
    model, model_args = _load_model(args.config, args.device)
    inputs = _make_inputs(
        model_args=model_args,
        n_res=args.n_res,
        prot_seq_len=args.prot_seq_len,
        na_seq_len=args.na_seq_len,
        map_size=args.map_size,
        device=args.device,
    )
    timers = TimerStore(args.device)
    _attach_timers(model, timers)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(**inputs, run_iters=args.run_iters)

        for _ in range(args.steps):
            _ = model(**inputs, run_iters=args.run_iters)

    _print_results(timers, args.steps, args.topk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

