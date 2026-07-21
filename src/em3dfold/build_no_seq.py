"""Sequence-free de novo modeling entrypoint."""
import argparse
import os
import shutil
import time
from contextlib import contextmanager

from em3dfold import build as build_with_seq
from em3dfold.utils.log_utils import get_runtime_log_path, progress
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.utils.torch_utils import clear_cuda_cache, get_device_names


@contextmanager
def _temporary_chain_hmm_profile_dir(profile_dir):
    previous = os.environ.get("EM3DFOLD_CHAIN_HMM_PROFILE_DIR")
    if profile_dir is None:
        if previous is None:
            yield
            return
        del os.environ["EM3DFOLD_CHAIN_HMM_PROFILE_DIR"]
        try:
            yield
        finally:
            os.environ["EM3DFOLD_CHAIN_HMM_PROFILE_DIR"] = previous
        return

    os.environ["EM3DFOLD_CHAIN_HMM_PROFILE_DIR"] = str(profile_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EM3DFOLD_CHAIN_HMM_PROFILE_DIR", None)
        else:
            os.environ["EM3DFOLD_CHAIN_HMM_PROFILE_DIR"] = previous


def add_args(parser):
    parser.usage = (
        "%(prog)s [--verbose] --map MAP --output OUTPUT [--protein] [--rna] [--dna] [--all] "
        "[--device DEVICE] [--pred-weights-dir PRED_WEIGHTS_DIR] "
        "[--protein-all-atom-weights PROTEIN_ALL_ATOM_WEIGHTS] "
        "[--na-all-atom-weights NA_ALL_ATOM_WEIGHTS] [--na-aa-weights NA_AA_WEIGHTS] "
        "[--recycle RECYCLE] [--keep-temp-files]"
    )
    parser.add_argument("--map", "-m", help="Input map", required=True)
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument("--protein", action="store_true", help="Build only the protein part, or combine with --rna/--dna")
    parser.add_argument("--rna", action="store_true", help="Build the nucleic-acid part")
    parser.add_argument("--dna", action="store_true", help="Build the nucleic-acid part")
    parser.add_argument("--all", action="store_true", help="Build all supported polymer types; this is also the default when no selector is given")
    parser.add_argument(
        "--device",
        "--gpu",
        help="Compute device. Use a single device such as '0' or 'cpu', or a comma-separated GPU list such as '0,1,2,3'",
        default="0",
    )
    parser.add_argument(
        "--pred-weights-dir",
        "--weights-dir",
        dest="pred_weights_dir",
        help="Optional shared root directory for weights",
    )
    parser.add_argument(
        "--protein-all-atom-weights",
        help="Optional override for protein all-atom denovo weights; defaults to <weights>/protein/model_all_atom_no_lm",
    )
    parser.add_argument(
        "--na-all-atom-weights",
        help="Optional override for nucleic-acid all-atom denovo weights; defaults to <weights>/na/model_all_atom_no_lm",
    )
    parser.add_argument(
        "--cpx-all-atom-weights",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--na-aa-weights",
        help="Optional override for voxel-based nucleic-acid typing weights; defaults to <weights>/na/model_na_aa_new",
    )
    parser.set_defaults(infer_na_aa=True)
    parser.add_argument("--infer-na-aa", dest="infer_na_aa", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-infer-na-aa", dest="infer_na_aa", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--protein-model-config", help=argparse.SUPPRESS)
    parser.add_argument("--na-model-config", help=argparse.SUPPRESS)
    parser.add_argument("--cpx-model-config", help=argparse.SUPPRESS)
    parser.add_argument(
        "--recycle",
        type=int,
        default=None,
        help="Shared denovo recycle count. Also used by legacy/cpx fallback when set.",
    )
    parser.add_argument("--protein-recycle", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--na-recycle", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-per-residue", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--protein-repeat-per-residue", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--na-repeat-per-residue", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--temp-root", help=argparse.SUPPRESS)
    parser.add_argument("--use-system-temp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ca-component-link-distance", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--ca-component-min-size", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--ca-component-min-fraction-largest", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--keep-temp-files", "-k", action="store_true", help="Whether to keep temp files")
    parser.set_defaults(keep_hmm_files=True)
    parser.add_argument("--keep-hmm-files", dest="keep_hmm_files", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-keep-hmm-files", dest="keep_hmm_files", action="store_false", help=argparse.SUPPRESS)

    skip_group = parser.add_argument_group("Skipping options")
    skip_group.add_argument("--skip-preprocess", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-cx", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-denovo", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-map-to-p", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-protein", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na", action="store_true", help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na-aa", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_no_seq_build_targets(args):
    if bool(getattr(args, "all", False)) or not any(
        [bool(getattr(args, "protein", False)), bool(getattr(args, "rna", False)), bool(getattr(args, "dna", False))]
    ):
        return True, True, "all"

    run_protein = bool(getattr(args, "protein", False))
    run_nucleic = bool(getattr(args, "rna", False) or getattr(args, "dna", False))

    if run_protein and run_nucleic:
        return True, True, "protein+na"
    if run_protein:
        return True, False, "protein"
    return False, True, "na"


def _count_backbone_atoms(pdb_path, atom_name):
    if pdb_path is None or (not os.path.exists(pdb_path)):
        return 0
    count = 0
    with open(pdb_path, 'r') as handle:
        for line in handle:
            if not line.startswith('ATOM'):
                continue
            if line[12:16].strip() == atom_name:
                count += 1
    return count


def _run_infer_job_no_seq(*, chain_hmm_profile_dir=None, force_protein_mode=False, force_na_mode=False, **kwargs):
    from em3dfold.infer import infer

    inferlm_args = argparse.Namespace()
    inferlm_args.map = kwargs["map_path"]
    inferlm_args.polymer = kwargs["polymer_path"]
    inferlm_args.model_dir = kwargs["model_dir"]
    inferlm_args.device = kwargs["device"]
    inferlm_args.crop_length = 200
    inferlm_args.repeat_per_residue = int(kwargs.get("repeat_per_residue", 2))
    inferlm_args.run_iters = 3
    inferlm_args.batch_size = 1
    inferlm_args.fp16 = False
    inferlm_args.voxel_size = 1.0
    inferlm_args.refine = False
    inferlm_args.no_use_random_affine = False
    inferlm_args.recycle = kwargs["recycle"]
    inferlm_args.prot_seq_embed = None
    inferlm_args.na_seq_embed = None
    inferlm_args.na_aa_logits = kwargs.get("na_aa_logits")
    inferlm_args.output_dir = kwargs["output_dir"]
    inferlm_args.protein_seq = None
    inferlm_args.dna_seq = None
    inferlm_args.rna_seq = None
    inferlm_args.min_na_chain_len = 3
    inferlm_args.fallback_to_predicted_na_types = kwargs.get("fallback_to_predicted_na_types", True)
    inferlm_args.pass_prev_aa_probs = True
    inferlm_args.pass_prev_rmsd = True
    inferlm_args.pass_prev_node = True
    inferlm_args.model_config = kwargs["model_config"]
    inferlm_args.force_protein_mode = bool(force_protein_mode)
    inferlm_args.force_na_mode = bool(force_na_mode)
    print(f"# inferlm model weights: {inferlm_args.model_dir}")
    print(f"# inferlm model config: {inferlm_args.model_config}")
    print(f"# inferlm polymer input: {inferlm_args.polymer}")
    with _temporary_chain_hmm_profile_dir(chain_hmm_profile_dir):
        infer.main(inferlm_args)


def main(args):
    build_started_at = time.time()
    os.environ["EM3DFOLD_KEEP_HMM_FILES"] = "1" if bool(getattr(args, "keep_hmm_files", True)) else "0"
    script_dir = os.path.dirname(__file__)
    inferlm_no_seq_model_config = pjoin(
        script_dir, "infer", "config", "model_v3x2_12l_256_128_h8_no_lm.yaml"
    )
    inferlm_cpx_model_config = (
        build_with_seq._resolve_optional_file_path(getattr(args, "cpx_model_config", None))
        or inferlm_no_seq_model_config
    )
    inferlm_protein_model_config = (
        build_with_seq._resolve_optional_file_path(getattr(args, "protein_model_config", None))
        or inferlm_no_seq_model_config
    )
    inferlm_na_model_config = (
        build_with_seq._resolve_optional_file_path(getattr(args, "na_model_config", None))
        or inferlm_no_seq_model_config
    )

    weights_root_dir = build_with_seq._resolve_pred_weights_dir(args.pred_weights_dir, script_dir)
    pred_weights_dir = weights_root_dir
    dual_pred_weights_path = pjoin(weights_root_dir, "cpx", "model_dual")
    cpx_all_atom_weights_dir = (
        build_with_seq._resolve_optional_file_path(getattr(args, "cpx_all_atom_weights", None))
        or pjoin(weights_root_dir, "cpx", "model_all_atom_no_lm")
    )
    protein_all_atom_weights_dir = (
        build_with_seq._resolve_optional_file_path(getattr(args, "protein_all_atom_weights", None))
        or pjoin(weights_root_dir, "protein", "model_all_atom_no_lm")
    )
    na_all_atom_weights_dir = (
        build_with_seq._resolve_optional_file_path(getattr(args, "na_all_atom_weights", None))
        or pjoin(weights_root_dir, "na", "model_all_atom_no_lm")
    )
    na_aa_weights_path = (
        build_with_seq._resolve_optional_file_path(getattr(args, "na_aa_weights", None))
        or pjoin(weights_root_dir, "na", "model_na_aa_new")
    )
    enable_infer_na_aa = bool(getattr(args, "infer_na_aa", True)) and (not args.skip_infer_na_aa)

    out_dir = abspath(args.output)
    temp_dir = build_with_seq._prepare_temp_dir(out_dir, temp_root=args.temp_root, use_system_temp=args.use_system_temp)
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

    active_stages = ["preprocess", "pred", "denovo"]

    progress(f"EM3DFold begin at {build_with_seq._format_wall_time(build_started_at)}")
    for line in build_with_seq.BUILD_CONTACT_LINES:
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
    progress("Warning: current run is in no-seq mode; make sure this is what you want.")

    multi_stage_device = args.device
    single_stage_device = build_with_seq._primary_device(args.device)

    build_with_seq._announce_build_stage("preprocess", active_stages)
    if not args.skip_preprocess:
        start = time.time()
        from em3dfold.pipeline import preprocess
        preprocess_args = argparse.Namespace()
        preprocess_args.map = args.map
        preprocess_args.protein = None
        preprocess_args.rna = None
        preprocess_args.dna = None
        preprocess_args.output = temp_dir
        preprocess_args.device = single_stage_device
        preprocess.main(preprocess_args)
        print("# Time = {:.4f}".format(time.time() - start))
        build_with_seq._finish_build_stage(start)
    else:
        build_with_seq._finish_build_stage(skipped=True)

    runtime_map_path = build_with_seq._resolve_runtime_map_path(args, temp_dir)
    run_protein_input, run_nucleic_input, selected_mode = _resolve_no_seq_build_targets(args)
    print(f"# No-seq molecule mode: {selected_mode}")
    na_aa_logits_path = None

    build_with_seq._announce_build_stage("pred", active_stages)
    if not args.skip_cx:
        build_with_seq._emit_stage_runtime_hint("pred")
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
                na_aa_logits_path = build_with_seq._run_pred_na_type_job(
                    map_path=pjoin(temp_dir, "pred", "na.mrc"),
                    ckpt_path=na_aa_weights_path,
                    device=multi_stage_device,
                    output_dir=na_aa_output_dir,
                    stride=24,
                )
                clear_cuda_cache(multi_stage_device, note="pred_na_type")
                print(f"# build_no_seq: voxel NA typing logits = {na_aa_logits_path}")
            else:
                print("# build_no_seq: voxel NA typing disabled; use model prediction fallback only")
        print("# Time = {:.4f}".format(time.time() - start))
        build_with_seq._finish_build_stage(start)
    else:
        expected_na_aa_logits_path = pjoin(temp_dir, "pred", "na_aa", "logits.npz")
        if os.path.exists(expected_na_aa_logits_path):
            na_aa_logits_path = expected_na_aa_logits_path
            print(f"# Reuse voxel NA typing logits = {na_aa_logits_path}")
        build_with_seq._finish_build_stage(skipped=True)

    build_with_seq._announce_build_stage("denovo", active_stages)
    if not args.skip_denovo:
        stage_start = time.time()
        build_with_seq._emit_stage_runtime_hint("denovo")
        if not args.skip_map_to_p:
            if run_nucleic_input:
                start = time.time()
                raw_c4_path = pjoin(temp_dir, "pred", "raw_c4.pdb")
                try:
                    build_with_seq._run_getp_pipeline(
                        map_path=pjoin(temp_dir, "pred", "c4.mrc"),
                        output_dir=pjoin(temp_dir, "pred", "c4_getp"),
                        device=single_stage_device,
                        raw_output_path=raw_c4_path,
                        atom_name="C4'",
                        res_name="A",
                        chain_id="B",
                        element="C",
                        rmax=10.0,
                        dmerge=3.0,
                        thresh=8.0,
                        run_getp=True,
                        run_g2p=False,
                        g2p_neighbor_distance_threshold=0.0,
                    )
                except Exception as exc:
                    if os.path.exists(raw_c4_path):
                        os.remove(raw_c4_path)
                    print(f"# build_no_seq: skip NA seed extraction because C4' tracing failed: {exc}")
                print("# Time = {:.4f}".format(time.time() - start))
            if run_protein_input:
                start = time.time()
                raw_ca_path = pjoin(temp_dir, "pred", "raw_ca.pdb")
                try:
                    build_with_seq._run_getp_pipeline(
                        map_path=pjoin(temp_dir, "pred", "ca.mrc"),
                        output_dir=pjoin(temp_dir, "pred", "ca_getp"),
                        device=single_stage_device,
                        raw_output_path=raw_ca_path,
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
                except Exception as exc:
                    if os.path.exists(raw_ca_path):
                        os.remove(raw_ca_path)
                    print(f"# build_no_seq: skip protein seed extraction because CA tracing failed: {exc}")
                print("# Time = {:.4f}".format(time.time() - start))
        else:
            print("# Skip map to p")

        run_protein = run_protein_input and (not args.skip_infer_protein)
        run_na = run_nucleic_input and (not args.skip_infer_na)
        raw_ca_path = pjoin(temp_dir, "pred", "raw_ca.pdb")
        raw_c4_path = pjoin(temp_dir, "pred", "raw_c4.pdb")
        protein_seed_count = _count_backbone_atoms(raw_ca_path, "CA") if run_protein else 0
        na_seed_count = _count_backbone_atoms(raw_c4_path, "C4'") if run_na else 0
        if run_protein and (not os.path.exists(raw_ca_path)):
            print("# build_no_seq: skip protein denovo because raw_ca.pdb is missing")
            run_protein = False
        elif run_protein and protein_seed_count < 2:
            print(f"# build_no_seq: skip protein denovo because only {protein_seed_count} CA seed(s) were detected")
            run_protein = False
        if run_na and (not os.path.exists(raw_c4_path)):
            print("# build_no_seq: skip NA denovo because raw_c4.pdb is missing")
            run_na = False
        elif run_na and na_seed_count < 3:
            print(f"# build_no_seq: skip NA denovo because only {na_seed_count} C4' seed(s) were detected")
            run_na = False
        if args.skip_infer_na_aa:
            print("# skip-infer-na-aa is deprecated; treated as --no-infer-na-aa")

        if run_protein or run_na:
            denovo_dir = pjoin(temp_dir, "denovo")
            os.makedirs(denovo_dir, exist_ok=True)
            print("# No-seq mode: skip protein LM embedding")
            print("# No-seq mode: skip nucleic-acid LM embedding")
            initial_polymer_path = pjoin(temp_dir, "pred", "raw_polymer.pdb")
            build_with_seq._merge_initial_polymer_files(
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

            cpx_recycle, protein_recycle, na_recycle = build_with_seq._resolve_denovo_recycles(args, run_protein=run_protein, run_na=run_na)
            cpx_repeat_per_residue, protein_repeat_per_residue, na_repeat_per_residue = build_with_seq._resolve_denovo_repeat_per_residues(args)
            print("# Recycle settings: cpx={} protein={} na={}".format(cpx_recycle, protein_recycle, na_recycle))
            print("# Repeat-per-residue settings: cpx={} protein={} na={}".format(cpx_repeat_per_residue, protein_repeat_per_residue, na_repeat_per_residue))

            start = time.time()
            if len(split_model_missing) == 0:
                print("# Denovo mode: split protein/NA all-atom models (no-seq)")
                protein_denovo_dir = pjoin(denovo_dir, "protein") if run_protein else None
                na_denovo_dir = pjoin(denovo_dir, "na") if run_na else None
                if run_protein:
                    os.makedirs(protein_denovo_dir, exist_ok=True)
                    _run_infer_job_no_seq(
                        map_path=runtime_map_path,
                        polymer_path=pjoin(temp_dir, "pred", "raw_ca.pdb"),
                        model_dir=protein_all_atom_weights_dir,
                        model_config=inferlm_protein_model_config,
                        device=multi_stage_device,
                        output_dir=protein_denovo_dir,
                        recycle=protein_recycle,
                        repeat_per_residue=protein_repeat_per_residue,
                        fallback_to_predicted_na_types=False,
                        chain_hmm_profile_dir=pjoin(out_dir, "hmm_profiles", "protein"),
                        force_protein_mode=True,
                        force_na_mode=False,
                    )
                    clear_cuda_cache(multi_stage_device, note="infer_protein_no_seq")
                if run_na:
                    os.makedirs(na_denovo_dir, exist_ok=True)
                    _run_infer_job_no_seq(
                        map_path=runtime_map_path,
                        polymer_path=pjoin(temp_dir, "pred", "raw_c4.pdb"),
                        model_dir=na_all_atom_weights_dir,
                        model_config=inferlm_na_model_config,
                        device=multi_stage_device,
                        output_dir=na_denovo_dir,
                        recycle=na_recycle,
                        repeat_per_residue=na_repeat_per_residue,
                        na_aa_logits=na_aa_logits_path,
                        fallback_to_predicted_na_types=True,
                        chain_hmm_profile_dir=pjoin(out_dir, "hmm_profiles", "na"),
                        force_protein_mode=False,
                        force_na_mode=True,
                    )
                    clear_cuda_cache(multi_stage_device, note="infer_na_no_seq")
                if run_protein and run_na:
                    merged_output_path, merged_entropy_output_path = build_with_seq._merge_split_denovo_outputs(
                        denovo_dir,
                        protein_output_dir=protein_denovo_dir,
                        na_output_dir=na_denovo_dir,
                    )
                else:
                    source_output_dir = protein_denovo_dir if run_protein else na_denovo_dir
                    merged_output_path = None
                    merged_entropy_output_path = None
                    if source_output_dir is not None:
                        source_output_path = build_with_seq._first_existing_path(
                            pjoin(source_output_dir, "output.cif"),
                        )
                        if source_output_path is not None:
                            merged_output_path = pjoin(denovo_dir, "output.cif")
                            shutil.copy(source_output_path, merged_output_path)
                        source_entropy_path = build_with_seq._first_existing_path(
                            pjoin(source_output_dir, "output_entropy_score.cif"),
                        )
                        if source_entropy_path is not None:
                            merged_entropy_output_path = pjoin(denovo_dir, "output_entropy_score.cif")
                            shutil.copy(source_entropy_path, merged_entropy_output_path)
                print(f"# Merged denovo output: {merged_output_path}")
                print(f"# Merged denovo entropy output: {merged_entropy_output_path}")
            else:
                missing_str = ", ".join(f"{name}={path}" for name, path in split_model_missing)
                print(f"# Denovo mode: fallback to legacy cpx model because split weights are missing: {missing_str}")
                _run_infer_job_no_seq(
                    map_path=runtime_map_path,
                    polymer_path=initial_polymer_path,
                    model_dir=cpx_all_atom_weights_dir,
                    model_config=inferlm_cpx_model_config,
                    device=multi_stage_device,
                    output_dir=denovo_dir,
                    recycle=cpx_recycle,
                    repeat_per_residue=cpx_repeat_per_residue,
                    na_aa_logits=na_aa_logits_path,
                    fallback_to_predicted_na_types=bool(run_na),
                    chain_hmm_profile_dir=pjoin(out_dir, "hmm_profiles", "cpx"),
                    force_protein_mode=bool(run_protein),
                    force_na_mode=bool(run_na),
                )
                clear_cuda_cache(multi_stage_device, note="infer_cpx_no_seq")
            print("# Time = {:.4f}".format(time.time() - start))
        else:
            print("# Skip denovo")
        build_with_seq._finish_build_stage(stage_start)
    else:
        build_with_seq._finish_build_stage(skipped=True)

    final_denovo = build_with_seq._first_existing_path(
        pjoin(temp_dir, "denovo", "output.cif"),
        pjoin(temp_dir, "denovo", "denovo.cif"),
    )
    final_entropy = build_with_seq._first_existing_path(
        pjoin(temp_dir, "denovo", "output_entropy_score.cif"),
    )
    final_output = pjoin(out_dir, "output.cif")
    denovo_output = pjoin(out_dir, "output_denovo.cif")
    entropy_output = pjoin(out_dir, "output_denovo_entropy_scores.cif")
    has_output = final_denovo is not None and os.path.exists(final_denovo)
    if has_output:
        shutil.copy(final_denovo, final_output)
        shutil.copy(final_denovo, denovo_output)
        build_with_seq.fix_quotes(final_output)
        build_with_seq.fix_quotes(denovo_output)
    if final_entropy is not None and os.path.exists(final_entropy):
        shutil.copy(final_entropy, entropy_output)
        build_with_seq.fix_quotes(entropy_output)

    if args.keep_temp_files:
        print(f"# Keep temp dir = {temp_dir}")
    elif os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    progress("")
    if has_output:
        progress("Modeling complete. EM3DFold finished successfully.")
        progress("Thanks for waiting. Your model is ready.")
        progress(f"Final model: {final_output}")
    else:
        progress("EM3DFold did not produce a final model.")
        progress("Please check run.log for the stage that failed.")
    progress(f"EM3DFold end at {build_with_seq._format_wall_time()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
