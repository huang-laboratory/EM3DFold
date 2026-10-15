#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_ROOT="${1:-}"
MODEL_WEIGHTS_DIR="${DOWNLOAD_ROOT}/weights"
LM_WEIGHTS_DIR="${DOWNLOAD_ROOT}/lm_weights"
ENV_MANAGER="${EM3DFOLD_ENV_MANAGER:-conda}"
ENV_NAME="${EM3DFOLD_ENV_NAME:-em3dfold}"
ENV_PREFIX="${EM3DFOLD_ENV_PREFIX:-}"

# Override EM3DFOLD_WEIGHTS_URL if your hosted EM3DFold weights tarball uses a different path.
EM3DFOLD_WEIGHTS_URL="${EM3DFOLD_WEIGHTS_URL:-http://huanglab.phys.hust.edu.cn/EM3DFold/weights/weights_v1.2.tgz}"
RINALMO_WEIGHTS_URL="${RINALMO_WEIGHTS_URL:-https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt}"
ESM2_WEIGHTS_URL="${ESM2_WEIGHTS_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt}"
ESM2_CONTACT_REGRESSION_URL="${ESM2_CONTACT_REGRESSION_URL:-https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt}"

fail() {
  echo "$*" >&2
  exit 1
}

run_step() {
  local msg="$1"
  shift
  "$@" || fail "$msg"
}

function require_cmd() {
  local cmd=$1
  "$cmd" --help >/dev/null 2>&1 || fail "Failed to detect '$cmd', maybe '$cmd' is not installed?"
}

function usage() {
  echo "usage: bash scripts/download.sh [download_root]"
  echo
  echo "Downloaded files will be saved under:"
  echo "  ${MODEL_WEIGHTS_DIR}"
  echo "  ${LM_WEIGHTS_DIR}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || -z "${DOWNLOAD_ROOT}" ]]; then
  usage
  exit 1
fi

echo "Checking required commands"
require_cmd wget
require_cmd tar

mkdir -p "${DOWNLOAD_ROOT}"
DOWNLOAD_ROOT="$(cd "${DOWNLOAD_ROOT}" && pwd)"
MODEL_WEIGHTS_DIR="${DOWNLOAD_ROOT}/weights"
LM_WEIGHTS_DIR="${DOWNLOAD_ROOT}/lm_weights"

mkdir -p "${MODEL_WEIGHTS_DIR}"
mkdir -p "${LM_WEIGHTS_DIR}"

echo "1. Download EM3DFold pretrained weights"
cd "${MODEL_WEIGHTS_DIR}"
run_step "Failed at step 1, unable to download EM3DFold pretrained weights" \
  wget "${EM3DFOLD_WEIGHTS_URL}" -O weights.tgz
run_step "Failed at step 1, unable to extract EM3DFold pretrained weights" \
  tar -zxf weights.tgz
cd "${ROOT_DIR}"

echo "2. Download RiNALMo pretrained weights"
run_step "Failed at step 2, unable to download RiNALMo weights" \
  wget "${RINALMO_WEIGHTS_URL}" -O "${LM_WEIGHTS_DIR}/rinalmo_giga_pretrained.pt"

echo "3. Download ESM2 pretrained weights"
run_step "Failed at step 3, unable to download ESM2 weights" \
  wget "${ESM2_WEIGHTS_URL}" -O "${LM_WEIGHTS_DIR}/esm2_t33_650M_UR50D.pt"
run_step "Failed at step 3, unable to download ESM2 contact regression weights" \
  wget "${ESM2_CONTACT_REGRESSION_URL}" -O "${LM_WEIGHTS_DIR}/esm2_t33_650M_UR50D-contact-regression.pt"

echo "Done download"
echo "EM3DFold model weights: ${MODEL_WEIGHTS_DIR}"
echo "LM weights: ${LM_WEIGHTS_DIR}"

if [[ "${ENV_MANAGER}" == */* ]]; then
  ENV_MANAGER_AVAILABLE=false
  if [[ -x "${ENV_MANAGER}" ]]; then
    ENV_MANAGER_AVAILABLE=true
  fi
else
  ENV_MANAGER_AVAILABLE=false
  if command -v "${ENV_MANAGER}" >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v "${ENV_MANAGER}")"
    ENV_MANAGER_AVAILABLE=true
  fi
fi

if [[ "${ENV_MANAGER_AVAILABLE}" == true ]]; then
  if [[ -n "${ENV_PREFIX}" ]]; then
    env_target=(--prefix "${ENV_PREFIX}")
    target_description="prefix '${ENV_PREFIX}'"
  else
    env_target=(--name "${ENV_NAME}")
    target_description="environment '${ENV_NAME}'"
  fi
  echo "4. Set EM_WEIGHTS_DIR in ${target_description}"
  run_step "Failed to set EM_WEIGHTS_DIR in ${target_description}" \
    "${ENV_MANAGER}" env config vars set "EM_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR}" "${env_target[@]}"
  echo "Set environment variable: EM_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR} in ${target_description}"
  echo "Re-activate the environment for the change to take effect."
else
  echo "Skip setting EM_WEIGHTS_DIR: '${ENV_MANAGER}' is not available"
  echo "Run manually later:"
  if [[ -n "${ENV_PREFIX}" ]]; then
    echo "> ${ENV_MANAGER} env config vars set EM_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR} --prefix ${ENV_PREFIX}"
  else
    echo "> ${ENV_MANAGER} env config vars set EM_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR} --name ${ENV_NAME}"
  fi
fi
