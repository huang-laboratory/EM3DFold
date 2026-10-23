#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="em3dfold"

fail() {
  echo "$*" >&2
  exit 1
}

run_step() {
  local msg="$1"
  shift
  "$@" || fail "$msg"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found in PATH: $cmd"
}

if [[ $# -gt 0 ]]; then
  fail "This script no longer accepts a conda path. Usage: bash scripts/install.sh"
fi

require_cmd conda
eval "$(conda shell.bash hook)"

if conda env list | awk -v name="${ENV_NAME}" '$1 == name { found = 1 } END { exit !found }'; then
  echo "1 Update existing conda environment ${ENV_NAME}"
  run_step "Failed at step 1, unable to update conda environment ${ENV_NAME}" \
    conda env update --name "${ENV_NAME}" --file "${ROOT_DIR}/env.yml"
else
  echo "1 Create conda environment ${ENV_NAME}"
  run_step "Failed at step 1, unable to create conda environment ${ENV_NAME}" \
    conda env create --name "${ENV_NAME}" --file "${ROOT_DIR}/env.yml"
fi

echo "2 Activate env"
run_step "Failed at step 2, unable to activate em3dfold env" \
  conda activate "${ENV_NAME}"

echo "3 Install flash attention"
run_step "Failed at step 3, unable to install flash attention" \
  bash "${ROOT_DIR}/flash_attn_whl/install_flash_attn.sh"
echo "Done install base env"

echo "4 Install EM3DFold"
run_step "Failed at step 4, unable to install EM3DFold" \
  python -m pip install "${ROOT_DIR}"
echo "Done install EM3DFold"

# Check installation
run_step "Failed at step 5, em3dfold --help did not run successfully" \
  python -m em3dfold --help
echo "Done check installation"
