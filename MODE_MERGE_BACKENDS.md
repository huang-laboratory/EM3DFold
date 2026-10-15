# Mode-merge backends

EM3DFold mode merging no longer relies on SciPy `cKDTree` by default. The
backend is selected with `EM3DFOLD_MODE_MERGE_BACKEND`:

- `auto` (default): exact CPU spatial-hash center selection plus blocked Torch
  membership on CUDA; full CPU spatial hash when CUDA is unavailable. If the
  automatic Torch membership fails at runtime, it falls back to CPU spatial.
- `spatial` (alias `cpu`): exact CPU spatial hash for selection and membership.
- `torch` (aliases `cuda`, `gpu`): exact CPU spatial-hash selection and blocked
  Torch GPU membership. CUDA is required when this is explicitly selected.
- `ckdtree`: legacy SciPy implementation. This is retained for comparisons,
  but is not selected automatically because it can segfault in the full
  pipeline on gu03.

The Torch query block size defaults to 256 and can be changed with
`EM3DFOLD_MODE_MERGE_BLOCK_SIZE`. The standalone mean-shift CLI exposes the
same controls as `--merge-backend` and `--merge-block-size`.

Both production backends preserve the original density sort, inclusive
`distance <= rmerge` suppression, and lowest-center-index rule for nearest
distance ties. Torch queries whose two best distances are within a
conservative floating-point error band are recomputed with NumPy.

Examples:

```bash
EM3DFOLD_MODE_MERGE_BACKEND=spatial python -m em3dfold build ...
EM3DFOLD_MODE_MERGE_BACKEND=torch python -m em3dfold build ...
EM3DFOLD_MODE_MERGE_BACKEND=auto python -m em3dfold build ...
```
