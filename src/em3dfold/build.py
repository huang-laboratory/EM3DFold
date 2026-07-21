"""Main program"""
import os
import sys
import time
import shutil
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import em3dfold

from em3dfold.io.pdbio import (
    chains_atom_pos_to_pdb,
    convert_to_chains,
    fix_quotes,
    read_pdb,
)
from em3dfold.io.seqio import read_fasta
from em3dfold.utils.log_utils import get_runtime_log_path, progress
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.utils.torch_utils import clear_cuda_cache, get_device_names

EM_WEIGHTS_ENV_VAR = "EM_WEIGHTS_DIR"
BUILD_CONTACT_LINES = (
    f"Version: {getattr(em3dfold, '__version__', 'unknown')}",
    "By Tao Li, Huang-lab, Huazhong University of Science and Technology",
)


def _collect_build_stage_order(template_chain_paths):
    stages = ["preprocess", "pred", "denovo"]
    if template_chain_paths:
        stages.extend(["fix", "imp", "fit", "assemble"])
    return stages


def _announce_build_stage(stage_name, active_stages):
    stage_idx = active_stages.index(stage_name) + 1
    progress("")
    progress(f"===== Stage {stage_idx}/{len(active_stages)}: {stage_name} =====")


def _finish_build_stage(start_time=None, *, skipped=False):
    elapsed = 0.0 if skipped or start_time is None else (time.time() - start_time)
    suffix = " (skipped)" if skipped else ""
    progress(f"Finished in {elapsed:.2f} seconds{suffix}")


def _emit_stage_runtime_hint(stage_name):
    if stage_name in {"pred"}:
        progress("This stage may take a few minutes if the target map is large.")
        progress("The runtime scales approx. linearly with map size: 300^3 voxels takes ~= 2 minutes")

    if stage_name in {"denovo"}:
        progress("This stage may take a few minutes if the target structure is large.")
        progress("The runtime scales approx. linearly with res. counts: 3000 res. takes ~= 2 minutes each round")

def _format_wall_time(timestamp=None):
    dt = datetime.now() if timestamp is None else datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def add_args(parser):
    parser.usage = (
        "%(prog)s [--verbose] --map MAP [--protein PROTEIN] [--rna RNA] [--dna DNA] "
        "--output OUTPUT [--device DEVICE] [--lm-weights-dir LM_WEIGHTS_DIR] "
        "[--pred-weights-dir PRED_WEIGHTS_DIR] "
        "[--protein-all-atom-weights PROTEIN_ALL_ATOM_WEIGHTS] "
        "[--na-all-atom-weights NA_ALL_ATOM_WEIGHTS] [--na-aa-weights NA_AA_WEIGHTS] "
        "[--recycle RECYCLE] [--keep-temp-files]"
    )
    parser.add_argument("--map", "-m", help="Input map", required=True)
    # Using --protein/--dna/--rna instead of a consensus --seq
    parser.add_argument("--protein", "-p", help="Input protein sequence")
    parser.add_argument("--rna", "-r", help="Input rna sequence")
    parser.add_argument("--dna", "-d", help="Input dna sequence")
    # Add protein chain templates, e.g. predicted AlphaFold models
    parser.add_argument(
        "--protein-template",
        "-pt",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument(
        "--device",
        "--gpu",
        help="Compute device. Use a single device such as '0' or 'cpu', or a comma-separated GPU list such as '0,1,2,3'",
        default="0",
    )
    parser.add_argument(
        "--lm-weights-dir",
        help="Optional shared directory for ESM and RiNALMo weights",
    )
    parser.add_argument(
        "--pred-weights-dir",
        "--weights-dir",
        dest="pred_weights_dir",
        help="Optional shared root directory for weights",
    )
    parser.add_argument(
        "--protein-all-atom-weights",
        help="Optional override for protein all-atom denovo weights; defaults to <weights>/protein/model_all_atom",
    )
    parser.add_argument(
        "--na-all-atom-weights",
        help="Optional override for nucleic-acid all-atom denovo weights; defaults to <weights>/na/model_all_atom",
    )
    parser.add_argument(
        "--na-aa-weights",
        help="Optional override for voxel-based nucleic-acid typing weights; defaults to <weights>/na/model_na_aa_new",
    )
    parser.set_defaults(infer_na_aa=True)
    parser.add_argument(
        "--infer-na-aa",
        dest="infer_na_aa",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-infer-na-aa",
        dest="infer_na_aa",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protein-model-config",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--na-model-config",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpx-model-config",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--recycle",
        type=int,
        default=None,
        help="Shared denovo recycle count. Also used by legacy/cpx fallback when set.",
    )
    parser.add_argument(
        "--protein-recycle",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--na-recycle",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repeat-per-residue",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protein-repeat-per-residue",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--na-repeat-per-residue",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--temp-root",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-system-temp",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ca-component-link-distance",
        type=float,
        default=6.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ca-component-min-size",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ca-component-min-fraction-largest",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--keep-temp-files", "-k", action="store_true", help="Whether to keep temp files")
    parser.set_defaults(keep_hmm_files=True)
    parser.add_argument(
        "--keep-hmm-files",
        dest="keep_hmm_files",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-keep-hmm-files",
        dest="keep_hmm_files",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    # Skipping controls
    skip_group = parser.add_argument_group("Skipping options")
    skip_group.add_argument("--skip-preprocess", action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-cx",         action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-denovo",     action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-map-to-p",   action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-protein", action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na",      action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na-aa",   action='store_true', help=argparse.SUPPRESS)
    # With templates
    skip_group.add_argument("--skip-imp",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-fix",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-fit",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-assemble", action='store_true', help=argparse.SUPPRESS)
    return parser


def _run_getp_pipeline(
    map_path,
    output_dir,
    device,
    raw_output_path,
    *,
    atom_name="CA",
    res_name="GLY",
    chain_id="A",
    element=None,
    rmax,
    dmerge,
    thresh,
    ratio=0.05,
    run_getp=True,
    run_g2p=False,
    g2p_neighbor_distance_threshold=6.0,
    component_link_distance=0.0,
    component_min_size=0,
    component_min_fraction_largest=0.0,
):
    if component_link_distance > 0.0 and (component_min_size > 0 or component_min_fraction_largest > 0.0):
        from em3dfold.pipeline import getp_filter as getp_module
    else:
        from em3dfold.pipeline import getp as getp_module

    os.makedirs(output_dir, exist_ok=True)
    getp_args = argparse.Namespace()
    getp_args.map = map_path
    getp_args.device = device
    getp_args.p = None
    getp_args.output = output_dir
    getp_args.getp = None
    getp_args.run_g2p = run_g2p
    getp_args.run_getp = run_getp
    getp_args.ratio = ratio
    getp_args.thresh = thresh
    getp_args.res = 6.0
    getp_args.nt = 4
    getp_args.filter = 0.0
    getp_args.dmerge = dmerge
    getp_args.rmax = rmax
    getp_args.fuse_g2p = True
    getp_args.g2p_neighbor_distance_threshold = g2p_neighbor_distance_threshold
    getp_args.g2p_cover_distance = 1.75
    getp_args.g2p_supplement_merge_distance = 1.5
    getp_args.g2p_refine_radius = 1.5
    getp_args.g2p_refine_iters = 2
    getp_args.g2p_max_supplements = None
    getp_args.atom_name = atom_name
    getp_args.res_name = res_name
    getp_args.chain_id = chain_id
    getp_args.element = element
    if component_link_distance > 0.0 and (component_min_size > 0 or component_min_fraction_largest > 0.0):
        getp_args.component_link_distance = component_link_distance
        getp_args.component_min_size = component_min_size
        getp_args.component_min_fraction_largest = component_min_fraction_largest
    getp_module.main(getp_args)

    merged_output_path = pjoin(output_dir, "merged.pdb")
    if os.path.exists(merged_output_path):
        shutil.copy(merged_output_path, raw_output_path)
    elif os.path.exists(pjoin(output_dir, "raw.pdb")):
        shutil.copy(pjoin(output_dir, "raw.pdb"), raw_output_path)
    else:
        raise FileNotFoundError(
            "getp did not produce merged.pdb or raw.pdb under {}".format(output_dir)
        )


def _first_existing_path(*paths):
    for path in paths:
        if path is not None and os.path.exists(path):
            return path
    return None


def _merge_initial_polymer_files(output_path, file_specs):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        atom_serial = 1
        for file_path, chain_id in file_specs:
            if file_path is None or (not os.path.exists(file_path)):
                continue
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.startswith(("ATOM", "HETATM")):
                        line = f"{line[:6]}{atom_serial:>5d}{line[11:]}"
                        line = f"{line[:21]}{chain_id}{line[22:]}"
                        fout.write(line)
                        atom_serial += 1
                    elif line.startswith("TER"):
                        fout.write("TER\n")
        fout.write("END\n")


def _normalize_torch_device(device):
    device = str(device)
    if device.isdigit():
        return f"cuda:{device}"
    return device


def _primary_device(device):
    return get_device_names(device)[0]


def _resolve_env_weights_root():
    env_value = os.environ.get(EM_WEIGHTS_ENV_VAR)
    if env_value is None or str(env_value).strip() == "":
        return None
    return Path(env_value).expanduser().resolve()


def _first_existing_path_obj(*paths):
    for path in paths:
        if path is not None and path.exists():
            return path.resolve()
    return None


def _resolve_pred_weights_dir(pred_weights_dir, script_dir):
    if pred_weights_dir:
        return abspath(pred_weights_dir)

    env_root = _resolve_env_weights_root()
    if env_root is not None:
        resolved = _first_existing_path_obj(
            env_root / "weights",
            env_root,
            env_root.parent / "weights" if env_root.name == "lm_weights" else None,
        )
        if resolved is not None:
            return str(resolved)
        if env_root.name == "weights":
            return str(env_root)
        return str((env_root / "weights").resolve())

    raise ValueError(
        "Prediction weights directory is not specified. "
        "Use --pred-weights-dir/--weights-dir or set EM_WEIGHTS_DIR."
    )


def _resolve_lm_weights_dir(lm_weights_dir):
    if lm_weights_dir:
        return Path(lm_weights_dir).expanduser().resolve()

    env_root = _resolve_env_weights_root()
    if env_root is None:
        return None

    resolved = _first_existing_path_obj(
        env_root / "lm_weights",
        env_root if env_root.name == "lm_weights" else None,
        env_root.parent / "lm_weights" if env_root.name == "weights" else None,
    )
    if resolved is not None:
        return resolved

    if env_root.name == "weights":
        return (env_root.parent / "lm_weights").resolve()
    return (env_root / "lm_weights").resolve()


def _resolve_optional_file_path(path):
    if path is None:
        return None
    return abspath(path)


def _run_infer_job(
    *,
    map_path,
    polymer_path,
    model_dir,
    model_config,
    device,
    output_dir,
    recycle,
    repeat_per_residue=2,
    protein_seq=None,
    dna_seq=None,
    rna_seq=None,
    prot_seq_embed=None,
    na_seq_embed=None,
    na_aa_logits=None,
    fallback_to_predicted_na_types=True,
):
    from em3dfold.infer import infer

    inferlm_args = argparse.Namespace()
    inferlm_args.map = map_path
    inferlm_args.polymer = polymer_path
    inferlm_args.model_dir = model_dir
    inferlm_args.device = device
    inferlm_args.crop_length = 200
    inferlm_args.repeat_per_residue = int(repeat_per_residue)
    inferlm_args.run_iters = 3
    inferlm_args.batch_size = 1
    inferlm_args.fp16 = False
    inferlm_args.voxel_size = 1.0
    inferlm_args.refine = False
    inferlm_args.no_use_random_affine = False
    inferlm_args.recycle = recycle
    inferlm_args.prot_seq_embed = prot_seq_embed
    inferlm_args.na_seq_embed = na_seq_embed
    inferlm_args.na_aa_logits = na_aa_logits
    inferlm_args.output_dir = output_dir
    inferlm_args.protein_seq = protein_seq
    inferlm_args.dna_seq = dna_seq
    inferlm_args.rna_seq = rna_seq
    inferlm_args.min_na_chain_len = 3
    inferlm_args.fallback_to_predicted_na_types = fallback_to_predicted_na_types
    inferlm_args.pass_prev_aa_probs = True
    inferlm_args.pass_prev_rmsd = True
    inferlm_args.pass_prev_node = True
    inferlm_args.model_config = model_config
    print(f"# inferlm model weights: {model_dir}")
    print(f"# inferlm model config: {model_config}")
    print(f"# inferlm polymer input: {polymer_path}")
    infer.main(inferlm_args)


def _run_pred_na_type_job(
    *,
    map_path,
    ckpt_path,
    device,
    output_dir,
    batchsize=40,
    stride=16,
    box_size=48,
):
    from em3dfold.pipeline import pred_na_type

    pred_args = argparse.Namespace()
    pred_args.input = map_path
    pred_args.output = output_dir
    pred_args.ckpt = ckpt_path
    pred_args.device = device
    pred_args.batchsize = int(batchsize)
    pred_args.stride = int(stride)
    pred_args.box_size = int(box_size)
    pred_args.apix = 1.0
    pred_args.normalize_percentile = 99.999
    pred_args.gaussian_sigma = None
    pred_args.gaussian_weight = True
    pred_args.fp16 = False
    pred_args.write_mrc = False
    pred_na_type.main(pred_args)
    return pjoin(output_dir, "logits.npz")


def _merge_split_denovo_outputs(output_dir, *, protein_output_dir=None, na_output_dir=None):
    from em3dfold.infer.inferlm_common import _write_merged_best_so_far_output

    protein_output_path = _first_existing_path(
        pjoin(protein_output_dir, "output.cif") if protein_output_dir is not None else None,
    )
    na_output_path = _first_existing_path(
        pjoin(na_output_dir, "output.cif") if na_output_dir is not None else None,
    )
    merged_output_path = _write_merged_best_so_far_output(
        pjoin(output_dir, "output.cif"),
        protein_path=protein_output_path,
        na_path=na_output_path,
    )

    protein_entropy_path = _first_existing_path(
        pjoin(protein_output_dir, "output_entropy_score.cif") if protein_output_dir is not None else None,
    )
    na_entropy_path = _first_existing_path(
        pjoin(na_output_dir, "output_entropy_score.cif") if na_output_dir is not None else None,
    )
    merged_entropy_output_path = _write_merged_best_so_far_output(
        pjoin(output_dir, "output_entropy_score.cif"),
        protein_path=protein_entropy_path,
        na_path=na_entropy_path,
    )

    return merged_output_path, merged_entropy_output_path


def _resolve_denovo_recycles(args, run_protein, run_na):
    del run_na
    shared_recycle = getattr(args, "recycle", None)
    protein_recycle = getattr(args, "protein_recycle", None)
    na_recycle = getattr(args, "na_recycle", None)

    resolved_protein_recycle = (
        protein_recycle if protein_recycle is not None else (shared_recycle if shared_recycle is not None else 4)
    )
    resolved_na_recycle = (
        na_recycle if na_recycle is not None else (shared_recycle if shared_recycle is not None else 3)
    )
    resolved_cpx_recycle = (
        shared_recycle if shared_recycle is not None else (4 if run_protein else 3)
    )
    return resolved_cpx_recycle, resolved_protein_recycle, resolved_na_recycle


def _resolve_denovo_repeat_per_residues(args):
    shared_repeat = getattr(args, "repeat_per_residue", None)
    if shared_repeat is None:
        shared_repeat = 2
    protein_repeat = getattr(args, "protein_repeat_per_residue", None)
    if protein_repeat is None:
        protein_repeat = shared_repeat
    na_repeat = getattr(args, "na_repeat_per_residue", None)
    if na_repeat is None:
        na_repeat = shared_repeat
    resolved_cpx_repeat = shared_repeat
    return int(resolved_cpx_repeat), int(protein_repeat), int(na_repeat)


def _prepare_temp_dir(out_dir, temp_root=None, use_system_temp=False):
    if use_system_temp:
        parent_dir = abspath(temp_root) if temp_root else None
        temp_dir = tempfile.mkdtemp(prefix="em3dfold_", dir=parent_dir)
    else:
        if temp_root:
            temp_dir = pjoin(abspath(temp_root), os.path.basename(out_dir), "temp")
        else:
            temp_dir = pjoin(out_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
    return abspath(temp_dir)


def _extract_protein_template_chains(template_paths, temp_dir):
    import numpy as np

    template_dir = pjoin(temp_dir, "templates")
    if os.path.exists(template_dir):
        shutil.rmtree(template_dir)
    os.makedirs(template_dir, exist_ok=True)

    written_paths = []
    for template_idx, template_path in enumerate(template_paths):
        template_path = abspath(template_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Protein template file not found: {template_path}")

        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
            template_path,
            keep_valid=False,
            return_bfactor=True,
        )

        protein_mask = res_type < 20
        if not np.any(protein_mask):
            print(f"# No protein residues found in template {template_path}, skip")
            continue

        atom_pos = atom_pos[protein_mask]
        atom_mask = atom_mask[protein_mask]
        res_type = res_type[protein_mask]
        res_idx = res_idx[protein_mask]
        chain_idx = chain_idx[protein_mask]
        bfactor = bfactor[protein_mask]

        unique_chain_indices = np.unique(chain_idx)
        for chain_local_idx, source_chain_idx in enumerate(unique_chain_indices):
            chain_mask = chain_idx == source_chain_idx
            output_cif_path = pjoin(
                template_dir,
                f"template_{template_idx}_chain_{chain_local_idx}.cif",
            )
            output_pdb_path = pjoin(
                template_dir,
                f"template_{template_idx}_chain_{chain_local_idx}.pdb",
            )
            chains_atom_pos_to_pdb(
                output_cif_path,
                chains_atom_pos=[atom_pos[chain_mask]],
                chains_atom_mask=[atom_mask[chain_mask]],
                chains_res_type=[res_type[chain_mask]],
                chains_res_idx=[res_idx[chain_mask]],
                chains_idx=[0],
                chains_bfactor=[bfactor[chain_mask]],
                suffix="cif",
            )
            chains_atom_pos_to_pdb(
                output_pdb_path,
                chains_atom_pos=[atom_pos[chain_mask]],
                chains_atom_mask=[atom_mask[chain_mask]],
                chains_res_type=[res_type[chain_mask]],
                chains_res_idx=[res_idx[chain_mask]],
                chains_idx=[0],
                chains_bfactor=[bfactor[chain_mask]],
                suffix="pdb",
            )
            written_paths.append(output_pdb_path)
            print(f"# Write protein template chain to {output_cif_path}")
            print(f"# Write protein template chain to {output_pdb_path}")

    if written_paths:
        print(f"# Extracted {len(written_paths)} protein template chains into {template_dir}")
    else:
        print("# No protein template chains were extracted")
    return written_paths


def _split_structure_to_chains(structure_path, output_dir, *, protein_only=False, suffix="pdb"):
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        structure_path,
        keep_valid=False,
        return_bfactor=True,
    )

    if protein_only:
        protein_mask = res_type < 20
        if not np.any(protein_mask):
            return []
        atom_pos = atom_pos[protein_mask]
        atom_mask = atom_mask[protein_mask]
        res_type = res_type[protein_mask]
        res_idx = res_idx[protein_mask]
        chain_idx = chain_idx[protein_mask]
        bfactor = bfactor[protein_mask]

    (
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactor,
    ) = convert_to_chains(
        chain_idx,
        atom_pos,
        atom_mask,
        res_type,
        res_idx,
        bfactor,
    )

    output_paths = []
    for chain_local_idx in range(len(chains_atom_pos)):
        output_path = pjoin(output_dir, f"chain_{chain_local_idx}.{suffix}")
        chains_atom_pos_to_pdb(
            output_path,
            chains_atom_pos=[chains_atom_pos[chain_local_idx]],
            chains_atom_mask=[chains_atom_mask[chain_local_idx]],
            chains_res_type=[chains_res_type[chain_local_idx]],
            chains_res_idx=[chains_res_idx[chain_local_idx]],
            chains_idx=[0],
            chains_bfactor=[chains_bfactor[chain_local_idx]],
            suffix=suffix,
        )
        output_paths.append(output_path)
    return output_paths


def _has_cli_sequence_arg(input_seq_path):
    return input_seq_path is not None and str(input_seq_path).strip() != ""


def _resolve_nonempty_seq_path(cli_has_seq_arg, formatted_seq_path, label):
    if not cli_has_seq_arg:
        return None
    if not os.path.exists(formatted_seq_path):
        print(f"# {label} sequence file not found at {formatted_seq_path}, skip {label}")
        return None

    try:
        seqs = [seq.strip() for seq in read_fasta(formatted_seq_path) if seq.strip()]
    except Exception as exc:
        print(f"# Failed to read {label} sequence file {formatted_seq_path}: {exc}")
        return None

    if not seqs:
        print(f"# {label} sequence file is empty at {formatted_seq_path}, skip {label}")
        return None

    if sum(len(seq) for seq in seqs) <= 0:
        print(f"# {label} sequence length is zero at {formatted_seq_path}, skip {label}")
        return None

    return formatted_seq_path


def _resolve_runtime_seq_path(
    cli_input_path,
    formatted_seq_path,
    label,
    *,
    allow_original_fallback=False,
):
    cli_has_seq_arg = _has_cli_sequence_arg(cli_input_path)
    resolved = _resolve_nonempty_seq_path(cli_has_seq_arg, formatted_seq_path, label)
    if resolved is not None:
        return resolved

    if not (allow_original_fallback and cli_has_seq_arg):
        return None

    original_path = abspath(cli_input_path)
    if not os.path.exists(original_path):
        print(f"# {label} original sequence file not found at {original_path}, skip {label}")
        return None

    print(
        f"# {label} formatted sequence file is unavailable at {formatted_seq_path}, "
        f"fall back to original input {original_path}"
    )
    return original_path


def _resolve_runtime_map_path(args, temp_dir):
    formatted_map_path = pjoin(temp_dir, "format_map.mrc")
    if os.path.exists(formatted_map_path):
        return formatted_map_path

    if args.skip_preprocess:
        raise FileNotFoundError(
            "Cannot find preprocessed map at {} while --skip-preprocess is enabled. "
            "Run without --skip-preprocess, or reuse a workspace that already contains format_map.mrc.".format(
                formatted_map_path
            )
        )

    raise FileNotFoundError(
        "Cannot find preprocessed map at {} after preprocess.".format(formatted_map_path)
    )


def _load_esm_model(device, lm_weights_dir=None):
    import torch
    import esm

    shared_dir = _resolve_lm_weights_dir(lm_weights_dir)
    if shared_dir is not None:
        local_ckpt = shared_dir / "esm2_t33_650M_UR50D.pt"
        if local_ckpt.exists() and hasattr(esm.pretrained, "load_model_and_alphabet_local"):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(local_ckpt))
        else:
            # Fall back to torch hub cache layout under the shared directory.
            os.environ["TORCH_HOME"] = str(shared_dir)
            torch.hub.set_dir(str(shared_dir))
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = model.eval().to(device)
    return model, alphabet


def _load_rinalmo_model(device, lm_weights_dir=None):
    import importlib
    import torch

    local_rinalmo_dir = abspath(
        pjoin(os.path.dirname(__file__), "rinalmo", "RiNALMo-1.0")
    )
    if not os.path.isdir(local_rinalmo_dir):
        raise FileNotFoundError(
            f"Local RiNALMo directory not found: {local_rinalmo_dir}"
        )
    if local_rinalmo_dir not in sys.path:
        sys.path.insert(0, local_rinalmo_dir)

    pretrained = importlib.import_module("rinalmo.pretrained")
    shared_dir = _resolve_lm_weights_dir(lm_weights_dir)
    pretrained_weights_path = None
    if shared_dir is not None:
        candidate_paths = [
            shared_dir / "rinalmo_giga_pretrained.pt",
            shared_dir / "giga-v1.pt",
            shared_dir / "rinalmo" / "rinalmo_giga_pretrained.pt",
            shared_dir / "rinalmo" / "giga-v1.pt",
        ]
        pretrained_weights_path = next(
            (path for path in candidate_paths if path.exists()),
            None,
        )
        if pretrained_weights_path is None:
            pretrained.DEFAULT_CACHE_DIR = shared_dir / "rinalmo"
        else:
            config = pretrained.model_config("giga")
            model = pretrained.RiNALMo(config)
            alphabet = pretrained.Alphabet(**config["alphabet"])
            model.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu"))
            model = model.to(device)
            model.eval()
            return model, alphabet

    model, alphabet = pretrained.get_pretrained_model(model_name="giga-v1")
    model = model.to(device)
    model.eval()
    return model, alphabet


def _build_protein_lm(seq_path, output_path, device, max_chain_length=1000, lm_weights_dir=None):
    from em3dfold.pipeline import get_lm
    import numpy as np

    device = _normalize_torch_device(device)
    sequences = get_lm.read_fasta(seq_path)
    filtered_sequences = []
    for sequence in sequences:
        filtered = [ch for ch in sequence if ch in get_lm.prot_restype1 and ch != "1"]
        filtered = "".join(filtered)
        if len(filtered) > 2:
            filtered_sequences.append(filtered)

    if not filtered_sequences:
        lm_embeddings = np.zeros((3, 1280), dtype=np.float32)
    else:
        model, alphabet = _load_esm_model(device, lm_weights_dir=lm_weights_dir)
        batch_converter = alphabet.get_batch_converter()
        lm_embeddings = get_lm.get_lm_embeddings(
            model,
            batch_converter,
            filtered_sequences,
            max_chain_length=max_chain_length,
        )
    np.save(output_path, lm_embeddings)
    return output_path


def _build_na_lm(
    rna_seq_path,
    dna_seq_path,
    output_path,
    device,
    max_chain_length=1000,
    lm_weights_dir=None,
):
    import numpy as np

    from em3dfold.pipeline import get_lm_na
    device = _normalize_torch_device(device)
    model, alphabet = _load_rinalmo_model(device, lm_weights_dir=lm_weights_dir)

    all_embeddings = []
    if rna_seq_path is not None and os.path.exists(rna_seq_path):
        rna_sequences = get_lm_na.filter_seqs(get_lm_na.read_fasta(rna_seq_path))
        if rna_sequences:
            all_embeddings.append(
                get_lm_na.get_lm_embeddings(
                    lang_model=model,
                    alphabet=alphabet,
                    sequences=rna_sequences,
                    max_chain_length=max_chain_length,
                )
            )

    if dna_seq_path is not None and os.path.exists(dna_seq_path):
        dna_sequences = get_lm_na.filter_seqs(get_lm_na.read_fasta(dna_seq_path))
        dna_sequences = [seq.replace("T", "U") for seq in dna_sequences]
        if dna_sequences:
            all_embeddings.append(
                get_lm_na.get_lm_embeddings(
                    lang_model=model,
                    alphabet=alphabet,
                    sequences=dna_sequences,
                    max_chain_length=max_chain_length,
                )
            )

    if all_embeddings:
        lm_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        lm_embeddings = np.zeros((3, 1280), dtype=np.float32)
    np.save(output_path, lm_embeddings)
    return output_path

def main(args):
    build_started_at = time.time()
    os.environ["EM3DFOLD_KEEP_HMM_FILES"] = "1" if bool(getattr(args, "keep_hmm_files", True)) else "0"
    script_dir = os.path.dirname(__file__)
    inferlm_cpx_model_config = _resolve_optional_file_path(args.cpx_model_config) or pjoin(
        script_dir, "infer", "config", "model_v3x2_12l_256_128_h8.yaml"
    )
    inferlm_split_model_config = pjoin(
        script_dir, "infer", "config", "model_v3x2_12l_256_128_h8.yaml"
    )
    inferlm_protein_model_config = (
        _resolve_optional_file_path(args.protein_model_config) or inferlm_split_model_config
    )
    inferlm_na_model_config = (
        _resolve_optional_file_path(args.na_model_config) or inferlm_split_model_config
    )
    weights_root_dir = _resolve_pred_weights_dir(args.pred_weights_dir, script_dir)
    pred_weights_dir = weights_root_dir
    dual_pred_weights_path = pjoin(weights_root_dir, "cpx", "model_dual")
    cpx_all_atom_weights_dir = pjoin(weights_root_dir, "cpx", "model_all_atom")
    protein_all_atom_weights_dir = (
        _resolve_optional_file_path(args.protein_all_atom_weights)
        or pjoin(weights_root_dir, "protein", "model_all_atom")
    )
    na_all_atom_weights_dir = (
        _resolve_optional_file_path(args.na_all_atom_weights)
        or pjoin(weights_root_dir, "na", "model_all_atom")
    )
    na_aa_weights_path = (
        _resolve_optional_file_path(args.na_aa_weights)
        or pjoin(weights_root_dir, "na", "model_na_aa_new")
    )
    enable_infer_na_aa = bool(getattr(args, "infer_na_aa", True)) and (not args.skip_infer_na_aa)

    out_dir = abspath(args.output)
    temp_dir = _prepare_temp_dir(
        out_dir,
        temp_root=args.temp_root,
        use_system_temp=args.use_system_temp,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"# Pred weights dir: {pred_weights_dir}")
    print(f"# dual stage1 weights path: {dual_pred_weights_path}")
    print(f"# inferlm cpx weights dir: {cpx_all_atom_weights_dir}")
    print(f"# inferlm protein weights dir: {protein_all_atom_weights_dir}")
    print(f"# inferlm NA weights dir: {na_all_atom_weights_dir}")
    print(f"# voxel NA typing weights path: {na_aa_weights_path}")
    print(f"# voxel NA typing enabled: {enable_infer_na_aa}")
    print(f"# inferlm fallback cpx config: {inferlm_cpx_model_config}")
    print(f"# inferlm protein config: {inferlm_protein_model_config}")
    print(f"# inferlm NA config: {inferlm_na_model_config}")
    print(f"# Temp root dir: {temp_dir}")

    template_chain_paths = []
    if args.protein_template:
        template_chain_paths = _extract_protein_template_chains(args.protein_template, temp_dir)

    active_stages = _collect_build_stage_order(template_chain_paths)

    progress(f"EM3DFold begin at {_format_wall_time(build_started_at)}")
    for line in BUILD_CONTACT_LINES:
        progress(line)
    resolved_devices = get_device_names(args.device)
    progress(f"Devices: {resolved_devices} (n={len(resolved_devices)})")
    progress(f"Output: {out_dir}")
    progress(f"Temp dir: {temp_dir}")
    runtime_log_path = get_runtime_log_path()
    if runtime_log_path is not None:
        progress(f"Run log: {runtime_log_path}")
    progress(f"Keep temporary files: {bool(args.keep_temp_files)}")
    progress(f"Keep HMM files: {bool(getattr(args, 'keep_hmm_files', True))}")

    multi_stage_device = args.device
    single_stage_device = _primary_device(args.device)

    has_protein_arg = _has_cli_sequence_arg(args.protein)
    has_rna_arg = _has_cli_sequence_arg(args.rna)
    has_dna_arg = _has_cli_sequence_arg(args.dna)
    if not (has_protein_arg or has_rna_arg or has_dna_arg):
        raise ValueError(
            "Please provide at least one input sequence via --protein and/or --rna and/or --dna."
        )

    # preprocess
    _announce_build_stage("preprocess", active_stages)
    if not args.skip_preprocess:
        start = time.time()
        from em3dfold.pipeline import preprocess
        preprocess_args = argparse.Namespace()
        preprocess_args.map = args.map
        preprocess_args.protein = args.protein
        preprocess_args.rna = args.rna
        preprocess_args.dna = args.dna
        preprocess_args.output = temp_dir

        preprocess_args.device = single_stage_device

        preprocess.main(preprocess_args)
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
        _finish_build_stage(start)
    else:
        _finish_build_stage(skipped=True)

    runtime_map_path = _resolve_runtime_map_path(args, temp_dir)


    protein_seq_path = _resolve_runtime_seq_path(
        args.protein,
        pjoin(temp_dir, "format_seq_protein.fasta"),
        "Protein",
        allow_original_fallback=bool(args.skip_preprocess),
    )
    rna_seq_path = _resolve_runtime_seq_path(
        args.rna,
        pjoin(temp_dir, "format_seq_rna.fasta"),
        "RNA",
        allow_original_fallback=bool(args.skip_preprocess),
    )
    dna_seq_path = _resolve_runtime_seq_path(
        args.dna,
        pjoin(temp_dir, "format_seq_dna.fasta"),
        "DNA",
        allow_original_fallback=bool(args.skip_preprocess),
    )
    run_protein_input = protein_seq_path is not None
    run_nucleic_input = (rna_seq_path is not None) or (dna_seq_path is not None)
    na_aa_logits_path = None

    # run segmentation
    _announce_build_stage("pred", active_stages)
    if not args.skip_cx:
        _emit_stage_runtime_hint("pred")
        run_nucleic = run_nucleic_input
        run_protein = run_protein_input

        start = time.time()
        from em3dfold.pipeline import pred

        if not os.path.exists(dual_pred_weights_path):
            raise FileNotFoundError(f"Dual stage1 weights are not found: {dual_pred_weights_path}")

        pred_args = argparse.Namespace()
        pred_args.input = runtime_map_path
        pred_args.output = pjoin(temp_dir, "pred")
        pred_args.ckpt = dual_pred_weights_path
        pred_args.contour = 1e-6
        pred_args.batchsize = 40
        pred_args.device = multi_stage_device
        pred_args.stride = 24
        pred_args.box_size = 48
        pred_args.apix = 1.0
        pred_args.gaussian_sigma = None
        pred_args.gaussian_weight = True
        pred_args.save_npz = False
        pred_args.fp16 = False

        pred.main(pred_args)
        clear_cuda_cache(multi_stage_device, note="pred")

        if run_nucleic_input and (not args.skip_infer_na):
            if enable_infer_na_aa:
                if not os.path.exists(na_aa_weights_path):
                    raise FileNotFoundError(
                        (
                            "Voxel NA typing weights are not found: {}. "
                            "Use --na-aa-weights to override or --no-infer-na-aa to disable."
                        ).format(na_aa_weights_path)
                    )
                na_aa_output_dir = pjoin(temp_dir, "pred", "na_aa")
                os.makedirs(na_aa_output_dir, exist_ok=True)
                na_aa_logits_path = _run_pred_na_type_job(
                    map_path=pjoin(temp_dir, "pred", "na.mrc"),
                    ckpt_path=na_aa_weights_path,
                    device=multi_stage_device,
                    output_dir=na_aa_output_dir,
                    stride=24,
                )
                clear_cuda_cache(multi_stage_device, note="pred_na_type")
                print(f"# build: voxel NA typing logits = {na_aa_logits_path}")
            else:
                print("# build: voxel NA typing disabled; use model prediction fallback only")

        end = time.time()

        print("# Time = {:.4f}".format(end - start))
        _finish_build_stage(start)
    else:
        expected_na_aa_logits_path = pjoin(temp_dir, "pred", "na_aa", "logits.npz")
        if os.path.exists(expected_na_aa_logits_path):
            na_aa_logits_path = expected_na_aa_logits_path
            print(f"# Reuse voxel NA typing logits = {na_aa_logits_path}")
        _finish_build_stage(skipped=True)


    # ms
    _announce_build_stage("denovo", active_stages)
    if not args.skip_denovo:
        stage_start = time.time()
        _emit_stage_runtime_hint("denovo")
        if not args.skip_map_to_p:
            ############
            # Handle C4'
            ############
            if run_nucleic_input:
                start = time.time()
                _run_getp_pipeline(
                    map_path=pjoin(temp_dir, "pred", "c4.mrc"),
                    output_dir=pjoin(temp_dir, "pred", "c4_getp"),
                    device=single_stage_device,
                    raw_output_path=pjoin(temp_dir, "pred", "raw_c4.pdb"),
                    atom_name="C4'",
                    res_name=("A" if rna_seq_path is not None else "DA"),
                    chain_id="B",
                    element="C",
                    rmax=10.0,
                    dmerge=3.0,
                    thresh=8.0,
                    run_getp=True,
                    run_g2p=False,
                    g2p_neighbor_distance_threshold=0.0,
                )
                end = time.time()
                print("# Time = {:.4f}".format(end - start))


            ###############
            # handle Calpha
            ###############
            if run_protein_input:
                start = time.time()
                _run_getp_pipeline(
                    map_path=pjoin(temp_dir, "pred", "ca.mrc"),
                    output_dir=pjoin(temp_dir, "pred", "ca_getp"),
                    device=single_stage_device,
                    raw_output_path=pjoin(temp_dir, "pred", "raw_ca.pdb"),
                    atom_name="CA",
                    res_name="GLY",
                    chain_id="A",
                    element="C",
                    rmax=1.0,
                    dmerge=1.0,
                    thresh=15.0,
                    ratio=0.05,
                    run_getp=True,
                    run_g2p=True,
                    component_link_distance=float(args.ca_component_link_distance),
                    component_min_size=int(args.ca_component_min_size),
                    component_min_fraction_largest=float(args.ca_component_min_fraction_largest),
                )
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
        else:
            print("# Skip map to p")


        run_protein = run_protein_input and (not args.skip_infer_protein)
        run_na = run_nucleic_input and (not args.skip_infer_na)

        if args.skip_infer_na_aa:
            print("# skip-infer-na-aa is deprecated; treated as --no-infer-na-aa")

        if run_protein or run_na:
            denovo_dir = pjoin(temp_dir, "denovo")
            os.makedirs(denovo_dir, exist_ok=True)

            prot_seq_embed_path = None
            na_seq_embed_path = None

            if run_protein:
                start = time.time()
                prot_seq_embed_path = pjoin(denovo_dir, "prot_lm.npy")
                _build_protein_lm(
                    protein_seq_path,
                    prot_seq_embed_path,
                    device=single_stage_device,
                    max_chain_length=1000,
                    lm_weights_dir=args.lm_weights_dir,
                )
                clear_cuda_cache(single_stage_device, note="protein LM")
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
            else:
                print("# Skip protein LM embedding")

            if run_na:
                start = time.time()
                na_seq_embed_path = pjoin(denovo_dir, "na_lm.npy")
                _build_na_lm(
                    rna_seq_path,
                    dna_seq_path,
                    na_seq_embed_path,
                    device=single_stage_device,
                    max_chain_length=1000,
                    lm_weights_dir=args.lm_weights_dir,
                )
                clear_cuda_cache(single_stage_device, note="NA LM")
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
            else:
                print("# Skip nucleic-acid LM embedding")

            initial_polymer_path = pjoin(temp_dir, "pred", "raw_polymer.pdb")
            _merge_initial_polymer_files(
                initial_polymer_path,
                [
                    (pjoin(temp_dir, "pred", "raw_ca.pdb") if run_protein else None, "A"),
                    (pjoin(temp_dir, "pred", "raw_c4.pdb") if run_na else None, "B"),
                ],
            )

            split_model_missing = []
            if run_protein and not os.path.exists(protein_all_atom_weights_dir):
                split_model_missing.append(("protein", protein_all_atom_weights_dir))
            if run_na and not os.path.exists(na_all_atom_weights_dir):
                split_model_missing.append(("na", na_all_atom_weights_dir))

            cpx_recycle, protein_recycle, na_recycle = _resolve_denovo_recycles(
                args,
                run_protein=run_protein,
                run_na=run_na,
            )
            cpx_repeat_per_residue, protein_repeat_per_residue, na_repeat_per_residue = _resolve_denovo_repeat_per_residues(args)
            print(
                "# Recycle settings: cpx={} protein={} na={}".format(
                    cpx_recycle,
                    protein_recycle,
                    na_recycle,
                )
            )
            print(
                "# Repeat-per-residue settings: cpx={} protein={} na={}".format(
                    cpx_repeat_per_residue,
                    protein_repeat_per_residue,
                    na_repeat_per_residue,
                )
            )

            start = time.time()
            if len(split_model_missing) == 0:
                print("# Denovo mode: split protein/NA all-atom models")
                protein_denovo_dir = pjoin(denovo_dir, "protein") if run_protein else None
                na_denovo_dir = pjoin(denovo_dir, "na") if run_na else None

                if run_protein:
                    os.makedirs(protein_denovo_dir, exist_ok=True)
                    _run_infer_job(
                        map_path=runtime_map_path,
                        polymer_path=pjoin(temp_dir, "pred", "raw_ca.pdb"),
                        model_dir=protein_all_atom_weights_dir,
                        model_config=inferlm_protein_model_config,
                        device=multi_stage_device,
                        output_dir=protein_denovo_dir,
                        recycle=protein_recycle,
                        repeat_per_residue=protein_repeat_per_residue,
                        protein_seq=protein_seq_path,
                        prot_seq_embed=prot_seq_embed_path,
                        fallback_to_predicted_na_types=False,
                    )
                    clear_cuda_cache(multi_stage_device, note="infer_protein")

                if run_na:
                    os.makedirs(na_denovo_dir, exist_ok=True)
                    _run_infer_job(
                        map_path=runtime_map_path,
                        polymer_path=pjoin(temp_dir, "pred", "raw_c4.pdb"),
                        model_dir=na_all_atom_weights_dir,
                        model_config=inferlm_na_model_config,
                        device=multi_stage_device,
                        output_dir=na_denovo_dir,
                        recycle=na_recycle,
                        repeat_per_residue=na_repeat_per_residue,
                        dna_seq=dna_seq_path,
                        rna_seq=rna_seq_path,
                        na_seq_embed=na_seq_embed_path,
                        na_aa_logits=na_aa_logits_path,
                        fallback_to_predicted_na_types=True,
                    )
                    clear_cuda_cache(multi_stage_device, note="infer_na")

                merged_output_path, merged_entropy_output_path = _merge_split_denovo_outputs(
                    denovo_dir,
                    protein_output_dir=protein_denovo_dir,
                    na_output_dir=na_denovo_dir,
                )
                print(f"# Merged denovo output: {merged_output_path}")
                print(f"# Merged denovo entropy output: {merged_entropy_output_path}")
            else:
                missing_str = ", ".join(f"{name}={path}" for name, path in split_model_missing)
                print(f"# Denovo mode: fallback to legacy cpx model because split weights are missing: {missing_str}")
                _run_infer_job(
                    map_path=runtime_map_path,
                    polymer_path=initial_polymer_path,
                    model_dir=cpx_all_atom_weights_dir,
                    model_config=inferlm_cpx_model_config,
                    device=multi_stage_device,
                    output_dir=denovo_dir,
                    recycle=cpx_recycle,
                    repeat_per_residue=cpx_repeat_per_residue,
                    protein_seq=protein_seq_path,
                    dna_seq=dna_seq_path,
                    rna_seq=rna_seq_path,
                    prot_seq_embed=prot_seq_embed_path,
                    na_seq_embed=na_seq_embed_path,
                    na_aa_logits=na_aa_logits_path,
                    fallback_to_predicted_na_types=bool(run_na),
                )
                clear_cuda_cache(multi_stage_device, note="infer_cpx")
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        else:
            print("# Skip denovo")
        _finish_build_stage(stage_start)

    else:
        _finish_build_stage(skipped=True)

    final_denovo = _first_existing_path(
        pjoin(temp_dir, "denovo", "output.cif"),
        pjoin(temp_dir, "denovo", "denovo.cif"),
    )
    final_entropy = _first_existing_path(
        pjoin(temp_dir, "denovo", "output_entropy_score.cif"),
    )
    fo = pjoin(out_dir, "output.cif")
    fo_denovo = pjoin(out_dir, "output_denovo.cif")
    fo_denovo_entropy = pjoin(out_dir, "output_denovo_entropy_scores.cif")
    fo_fit = pjoin(out_dir, "output_fit.cif")

    if final_denovo is not None and os.path.exists(final_denovo):
        shutil.copy(final_denovo, fo_denovo)
        fix_quotes(fo_denovo)
    if final_entropy is not None and os.path.exists(final_entropy):
        shutil.copy(final_entropy, fo_denovo_entropy)
        fix_quotes(fo_denovo_entropy)

    fix_output_dir = pjoin(temp_dir, "fix")
    imp_output_dir = pjoin(temp_dir, "imp")
    fit_output_dir = pjoin(temp_dir, "fit")
    fit_total_path = None
    template_candidate_paths = []
    denovo_protein_chain_paths = []
    shared_context = None
    template_refine_skip_reason = None
    template_refine_ready = False

    if template_chain_paths:
        if protein_seq_path is None:
            template_refine_skip_reason = "Protein sequence is unavailable, skip template fix/imp"
        elif final_denovo is None or (not os.path.exists(final_denovo)):
            template_refine_skip_reason = "De novo protein model is unavailable, skip template fix/imp"
        else:
            protein_chain_dir = pjoin(temp_dir, "template_refine", "denovo_protein_chains")
            denovo_protein_chain_paths = _split_structure_to_chains(
                final_denovo,
                protein_chain_dir,
                protein_only=True,
                suffix="pdb",
            )
            if not denovo_protein_chain_paths:
                template_refine_skip_reason = "No protein chains were found in the de novo model, skip template fix/imp"
            else:
                from em3dfold.template.pipeline import fix_pipeline, imp_pipeline
                from em3dfold.template.pipeline.template_refine import build_template_refine_context

                shared_context_dir = pjoin(temp_dir, "template_refine", "shared_context")
                os.makedirs(shared_context_dir, exist_ok=True)
                shared_context = build_template_refine_context(
                    chain_paths=denovo_protein_chain_paths,
                    template_paths=template_chain_paths,
                    lib_dir=script_dir,
                    work_dir=shared_context_dir,
                    seq_path=protein_seq_path,
                    verbose=False,
                    debug=False,
                    prepare_domains_flag=True,
                )
                print(f"# Shared protein template count = {len(shared_context.templates)}")

                if len(shared_context.templates) == 0:
                    template_refine_skip_reason = "No valid protein templates remain after filtering, skip template fix/imp"
                else:
                    template_refine_ready = True

    if template_chain_paths:
        _announce_build_stage("fix", active_stages)
        if args.skip_fix:
            _finish_build_stage(skipped=True)
        elif not template_refine_ready:
            if template_refine_skip_reason:
                print(f"# {template_refine_skip_reason}")
            _finish_build_stage(skipped=True)
        else:
            start = time.time()
            fix_args = argparse.Namespace(
                seq=protein_seq_path,
                chain=denovo_protein_chain_paths,
                template=template_chain_paths,
                lib=script_dir,
                output=fix_output_dir,
                verbose=False,
                debug=False,
            )
            fix_pipeline.run_with_context(fix_args, shared_context)
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
            _finish_build_stage(start)

        _announce_build_stage("imp", active_stages)
        if args.skip_imp:
            _finish_build_stage(skipped=True)
        elif not template_refine_ready:
            if template_refine_skip_reason:
                print(f"# {template_refine_skip_reason}")
            _finish_build_stage(skipped=True)
        else:
            start = time.time()
            imp_args = argparse.Namespace(
                seq=protein_seq_path,
                chain=denovo_protein_chain_paths,
                template=template_chain_paths,
                lib=script_dir,
                output=imp_output_dir,
                verbose=False,
                debug=False,
            )
            imp_pipeline.run_with_context(imp_args, shared_context)
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
            _finish_build_stage(start)

        _announce_build_stage("fit", active_stages)
        if args.skip_fit:
            _finish_build_stage(skipped=True)
        else:
            fit_map_path = pjoin(temp_dir, "pred", "mc.mrc")
            if not os.path.exists(fit_map_path):
                raise FileNotFoundError(
                    "Template fit requires temp/pred/mc.mrc, but it was not found."
                )

            start = time.time()
            from em3dfold.template.pipeline import fit_pipeline as template_fit

            fit_args = argparse.Namespace()
            fit_args.protein_template = template_chain_paths
            fit_args.map = fit_map_path
            fit_args.output = fit_output_dir
            fit_args.resolution = 6.0
            fit_args.threshold = 15.0
            fit_args.threshold_ratio = 0.10
            fit_args.rshift = 1.0
            fit_args.rmerge = 1.0
            fit_args.rmsdcut1 = 2.5
            fit_args.rmsdcut2 = 5.0
            fit_args.rigid_nleast = 5
            fit_args.rigid_cutoff_score_early = -1.5
            fit_args.rigid_cutoff_score_late = -0.5
            fit_args.rigid_skip_short_residues = 50
            fit_args.device = single_stage_device
            fit_args.angle_step = 18.0
            fit_args.fgrid = 3.0
            fit_args.sgrid = 2.0
            fit_args.ntrans = 8
            fit_args.ntop = 10
            fit_summary = template_fit.main(fit_args)
            fit_total_path = fit_summary.get("fitted_total_path")
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
            _finish_build_stage(start)

    if fit_total_path is None:
        fit_total_path = _first_existing_path(
            pjoin(fit_output_dir, "fitted_total.cif"),
        )

    imp_trimmed = pjoin(imp_output_dir, "imp_chains_trimmed.cif")
    if os.path.exists(imp_trimmed):
        template_candidate_paths.append(imp_trimmed)
    fix_chains = pjoin(fix_output_dir, "fix_chains_templs.cif")
    if os.path.exists(fix_chains):
        template_candidate_paths.append(fix_chains)
    if fit_total_path is not None and os.path.exists(fit_total_path):
        template_candidate_paths.append(fit_total_path)

    assemble_output_dir = pjoin(temp_dir, "assemble")
    assembled_output_path = None
    has_template_candidates = len(template_candidate_paths) > 0
    assemble_candidate_paths = list(template_candidate_paths)
    if (not has_template_candidates) and final_denovo is not None and os.path.exists(final_denovo):
        assemble_candidate_paths.append(final_denovo)
    elif run_nucleic_input and final_denovo is not None and os.path.exists(final_denovo):
        assemble_candidate_paths.append(final_denovo)

    if template_chain_paths:
        _announce_build_stage("assemble", active_stages)
        if args.skip_assemble:
            _finish_build_stage(skipped=True)
        else:
            ca_map_path = _first_existing_path(pjoin(temp_dir, "pred", "ca.mrc"))
            if assemble_candidate_paths and ca_map_path is not None and os.path.exists(ca_map_path):
                start = time.time()
                from em3dfold.pipeline import assemble as chain_assemble

                assemble_args = argparse.Namespace(
                    structure_paths=assemble_candidate_paths,
                    ca_map_path=ca_map_path,
                    output=assemble_output_dir,
                    clash_threshold=0.10,
                    clash_distance=1.0,
                    clash_resolution=5.0,
                    map_percentile=99.9,
                    time_limit=120.0,
                    num_workers=4,
                    log_search_progress=False,
                )
                chain_assemble.main(assemble_args)
                assembled_output_path = pjoin(assemble_output_dir, "assemble.cif")
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
                _finish_build_stage(start)
            elif assemble_candidate_paths:
                print("# Skip assemble: CA map is unavailable")
                _finish_build_stage(skipped=True)
            else:
                print("# Skip assemble: no structures are available")
                _finish_build_stage(skipped=True)

    if assembled_output_path is None:
        assembled_output_path = _first_existing_path(
            pjoin(assemble_output_dir, "assemble.cif"),
        )

    final_output_path = None
    for candidate in [
        assembled_output_path,
        final_denovo,
        fit_total_path,
        imp_trimmed,
        fix_chains,
    ]:
        if candidate is not None and os.path.exists(candidate):
            final_output_path = candidate
            break

    has_output = False
    if final_output_path is not None and os.path.exists(final_output_path):
        shutil.copy(final_output_path, fo)
        fix_quotes(fo)
        has_output = True
    if fit_total_path is not None and os.path.exists(fit_total_path):
        shutil.copy(fit_total_path, fo_fit)
        fix_quotes(fo_fit)


    # Remove temp files
    if not args.keep_temp_files:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    if has_output:
        progress("")
        progress("Modeling complete. EM3DFold finished successfully.")
        progress("Thanks for waiting. Your model is ready.")
        progress("Final model: {}".format(fo))
        # if os.path.exists(fo_denovo):
        #     progress("De novo model: {}".format(fo_denovo))
        # if os.path.exists(fo_denovo_entropy):
        #     progress("De novo model with entropy scores: {}".format(fo_denovo_entropy))
        # if os.path.exists(fo_fit):
        #     progress("Template fit model: {}".format(fo_fit))
        progress(f"EM3DFold end at {_format_wall_time()}")
    else:
        progress("")
        progress("EM3DFold did not produce a final model.")
        progress("Please check run.log for the stage that failed.")
        progress(f"EM3DFold end at {_format_wall_time()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
