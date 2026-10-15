# Mode-merge validation

Date: 2026-10-15

Base: EM3DFold main commit
`103ca12358f327a9a8904d21d708726e760d3baa`.

## Correctness

- Standard-library unit suite: 4 tests passed on gu03 GPU2.
- Cases cover duplicates, negative coordinates, inclusive radius boundaries,
  density-filtered points, and equal-distance lowest-index ties.
- 250 additional randomized Torch-versus-NumPy fuzz cases were identical,
  including varied coordinate scales and injected equal-distance points.
- Captured 9ZY0 and 8YIH inputs were run through legacy cKDTree, spatial, and
  Torch. For each target, all three backends produced identical center,
  center-density, and membership SHA-256 hashes.

Captured membership hashes:

| Target | Points | Centers | Membership SHA-256 |
|---|---:|---:|---|
| 9ZY0 | 1,489 | 773 | `2f98f2c8a797567e98b188bcf1b2aaed386e559a8f464d289a8751716a6035dd` |
| 8YIH | 3,194 | 1,386 | `a62b801930eb4876527d397b8a71b2247011fe970a4ccd1165b4be9162be274d` |

## Performance

Synthetic paired-point data use float32 coordinates, `rmerge=1.0`, and 50%
as many centers as input points. CPU spatial selection is shared by both
production backends. These cases ran concurrently in fresh processes.

| Backend | 10k | 20k | 50k | 50k GPU reserved |
|---|---:|---:|---:|---:|
| CPU spatial | 0.601 s | 1.152 s | 2.778 s | n/a |
| CPU spatial + Torch membership | 0.448 s | 0.651 s | 1.685 s | 202 MiB |

All six benchmark center and membership hashes equal the NumPy/main
reference hashes.

## Full 9ZY0 runs

Two independent full runs used original FP32 convolution and physical GPUs 4
and 5:

| Backend | Status | Log wall span | Result |
|---|---:|---:|---|
| spatial | 0 | 394.319 s | success |
| torch | 0 | 381.786 s | success |

The two C4/CA mode-merge calls took 0.0032/0.0624 s with spatial and
0.0177/0.0303 s with Torch. The first call is too small to amortize GPU launch
overhead; Torch is faster for the larger CA merge.

Both logs contain the selected backend for the C4 and CA merges and end with
`Modeling complete. EM3DFold finished successfully.` Neither contains a
segfault, traceback, or error.

The new spatial output, new Torch output, and the previously validated
main-semantics reference output are byte-for-byte identical. Final CIF:

`bc937d97d33938f4b9b0d0b34f531d9afb6b13982a7932675ae021aa1a935d04`

Entropy-score CIF:

`629d77c2afc65c4cd5a17390eb7da1212c4449fee7b3f46e6da52b3f6e85f848`

Run outputs:

- `/home/taoli/data_gu03/tests/EM3DFold_mode_merge_validation_20261015/spatial/output`
- `/home/taoli/data_gu03/tests/EM3DFold_mode_merge_validation_20261015/torch/output`

The original main worktree and base conda environment were not modified.
