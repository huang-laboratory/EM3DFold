"""Instrument a real sequence-free build without changing its precision or crops.

Requires all normal pipeline dependencies and official weights. Reports inclusive
wall times; nested categories must not be added together. GPU boundaries are
synchronized for attribution, so this is a profiling run, not a speed benchmark.
"""
import argparse
import functools
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protein', action='store_true')
    parser.add_argument('--rna', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    report = {'scope': 'real build_no_seq, normal defaults; inclusive synchronized wall times',
              'map': args.map, 'weights': args.weights, 'records': records,
              'conv_fp16_env': os.environ.get('EM3DFOLD_CONV_FP16', '0'),
              'strict_fp32_env': os.environ.get('EM3DFOLD_STRICT_FP32', '0'),
              'backend_env': os.environ.get('EM3DFOLD_STRUCTURE_BACKEND', 'auto'),
              'status': 'running', 'gpu': torch.cuda.get_device_name(0)}
    def save():
        groups = {}
        for row in records:
            item = groups.setdefault(row['category'], {'seconds': 0, 'calls': 0})
            item['seconds'] += row['seconds']; item['calls'] += 1
        report['inclusive_groups'] = groups
        (output/'profile.json').write_text(json.dumps(report, indent=2))
    def wrap(owner, attribute, category):
        original = getattr(owner, attribute)
        @functools.wraps(original)
        def timed(*a, **kw):
            torch.cuda.synchronize()
            started = time.perf_counter()
            error = None
            try:
                return original(*a, **kw)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                torch.cuda.synchronize()
                row = {'category': category, 'seconds': time.perf_counter()-started}
                if error: row['error'] = error
                records.append(row);save()
        setattr(owner, attribute, timed)
    # Compatibility shim only in this runner: numpy 2 removed asfarray.
    if not hasattr(np, 'asfarray'):
        np.asfarray = lambda value, dtype=np.float64: np.asarray(value, dtype=dtype)
        report['numpy_asfarray_shim'] = True
    from em3dfold import build, build_no_seq
    from em3dfold.pipeline import preprocess, pred, pred_na_type, getp, getp_filter
    from em3dfold.infer import inferlm_common, gnn_inference_utils, denovo
    from em3dfold.models.v3x2.model import Model
    from em3dfold.utils import multi_gpu_wrapper, cryo_utils
    for owner, attribute, category in [
        (preprocess, 'main', 'preprocess'), (pred, 'main', 'voxel_prediction'),
        (pred_na_type, 'main', 'voxel_na_typing'),
        (build, '_run_getp_pipeline', 'seed_extraction'),
        (inferlm_common, 'run_inference_loop', 'structure_recycle'),
        (multi_gpu_wrapper, 'init_model', 'structure_model_load'),
        (multi_gpu_wrapper, 'send_dict_to_device', 'host_to_device'),
        (gnn_inference_utils, 'get_inference_data', 'crop_preparation'),
        (gnn_inference_utils, 'collate_nn_results', 'prediction_collation'),
        (denovo, 'final_results_align_to_sequence', 'structure_postprocess'),
        (Model, 'forward', 'structure_forward'),
        (cryo_utils, 'parse_map', 'map_parse_and_resample'),
    ]:
        wrap(owner, attribute, category)
    pipeline_parser = build_no_seq.add_args(argparse.ArgumentParser())
    command = ['--map', args.map, '--output', str(output/'build'), '--device', '0',
               '--pred-weights-dir', args.weights, '--keep-temp-files']
    if args.protein: command.append('--protein')
    if args.rna: command.append('--rna')
    pipeline_args = pipeline_parser.parse_args(command)
    report['arguments'] = vars(pipeline_args)
    save();started = time.perf_counter()
    try:
        build_no_seq.main(pipeline_args)
        report['status'] = 'completed'
        report['final_files'] = [str(p) for p in (output/'build').glob('*.cif')]
    except Exception:
        report['status'] = 'failed';report['exception'] = traceback.format_exc();raise
    finally:
        report['total_seconds'] = time.perf_counter()-started;save()


if __name__ == '__main__':
    main()
