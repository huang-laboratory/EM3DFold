#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_ROOT="${1:-}"
DOWNLOAD_SOURCE="official"
DEBUG_DOWNLOAD=false
MODEL_WEIGHTS_DIR="${DOWNLOAD_ROOT}/weights"
LM_WEIGHTS_DIR="${DOWNLOAD_ROOT}/lm_weights"
ENV_MANAGER="${EM3DFOLD_ENV_MANAGER:-conda}"
ENV_NAME="${EM3DFOLD_ENV_NAME:-em3dfold}"
ENV_PREFIX="${EM3DFOLD_ENV_PREFIX:-}"

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
  echo "usage: bash scripts/download.sh DOWNLOAD_ROOT [--source official|huanglab] [--debug]"
  echo
  echo "Download sources:"
  echo "  official   Original upstream sources (default)"
  echo "  huanglab   Huang Lab mirror for the three language-model weight files"
  echo
  echo "Options:"
  echo "  --debug    Append .debug to each language-model weight URL"
  echo "             and output filename (only valid with --source huanglab)"
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

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 && -n "$2" ]] || {
        usage
        fail "--source requires 'official' or 'huanglab'"
      }
      DOWNLOAD_SOURCE="$2"
      shift 2
      ;;
    --debug)
      DEBUG_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      fail "Unknown argument '$1'"
      ;;
  esac
done

case "${DOWNLOAD_SOURCE}" in
  official)
    DEFAULT_RINALMO_WEIGHTS_URL="https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt"
    DEFAULT_ESM2_WEIGHTS_URL="https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
    DEFAULT_ESM2_CONTACT_REGRESSION_URL="https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt"
    ;;
  huanglab)
    HUANGLAB_LM_WEIGHTS_URL="http://huanglab.phys.hust.edu.cn/EM3DFold/lm_weights"
    DEFAULT_RINALMO_WEIGHTS_URL="${HUANGLAB_LM_WEIGHTS_URL}/rinalmo_giga_pretrained.pt"
    DEFAULT_ESM2_WEIGHTS_URL="${HUANGLAB_LM_WEIGHTS_URL}/esm2_t33_650M_UR50D.pt"
    DEFAULT_ESM2_CONTACT_REGRESSION_URL="${HUANGLAB_LM_WEIGHTS_URL}/esm2_t33_650M_UR50D-contact-regression.pt"
    ;;
  *)
    usage
    fail "Unknown download source '${DOWNLOAD_SOURCE}'. Expected 'official' or 'huanglab'."
    ;;
esac

if [[ "${DEBUG_DOWNLOAD}" == true && "${DOWNLOAD_SOURCE}" != "huanglab" ]]; then
  fail "--debug can only be used with '--source huanglab'"
fi

# When overriding a URL, also override its matching *_SHA256 variable.
EM3DFOLD_WEIGHTS_URL="${EM3DFOLD_WEIGHTS_URL:-http://huanglab.phys.hust.edu.cn/EM3DFold/weights/weights_v1.2.tgz}"
RINALMO_WEIGHTS_URL="${RINALMO_WEIGHTS_URL:-${DEFAULT_RINALMO_WEIGHTS_URL}}"
ESM2_WEIGHTS_URL="${ESM2_WEIGHTS_URL:-${DEFAULT_ESM2_WEIGHTS_URL}}"
ESM2_CONTACT_REGRESSION_URL="${ESM2_CONTACT_REGRESSION_URL:-${DEFAULT_ESM2_CONTACT_REGRESSION_URL}}"

DEFAULT_EM3DFOLD_WEIGHTS_SHA256="a9ab7a65020bcccfceddc0e4df8e300c0b1155fafc6c173af78785ed8ed13ede"
DEFAULT_RINALMO_WEIGHTS_SHA256="cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db"
DEFAULT_ESM2_WEIGHTS_SHA256="ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c"
DEFAULT_ESM2_CONTACT_REGRESSION_SHA256="8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575"

if [[ "${DEBUG_DOWNLOAD}" == true ]]; then
  RINALMO_WEIGHTS_URL="${RINALMO_WEIGHTS_URL}.debug"
  ESM2_WEIGHTS_URL="${ESM2_WEIGHTS_URL}.debug"
  ESM2_CONTACT_REGRESSION_URL="${ESM2_CONTACT_REGRESSION_URL}.debug"
  DEFAULT_RINALMO_WEIGHTS_SHA256="6562a35496cd7a03fe4fb23532b66fa84a4e4ef496d1af5477c940e037014c49"
  DEFAULT_ESM2_WEIGHTS_SHA256="${DEFAULT_RINALMO_WEIGHTS_SHA256}"
  DEFAULT_ESM2_CONTACT_REGRESSION_SHA256="${DEFAULT_RINALMO_WEIGHTS_SHA256}"
fi

RINALMO_WEIGHTS_FILENAME="rinalmo_giga_pretrained.pt"
ESM2_WEIGHTS_FILENAME="esm2_t33_650M_UR50D.pt"
ESM2_CONTACT_REGRESSION_FILENAME="esm2_t33_650M_UR50D-contact-regression.pt"
if [[ "${DEBUG_DOWNLOAD}" == true ]]; then
  RINALMO_WEIGHTS_FILENAME="${RINALMO_WEIGHTS_FILENAME}.debug"
  ESM2_WEIGHTS_FILENAME="${ESM2_WEIGHTS_FILENAME}.debug"
  ESM2_CONTACT_REGRESSION_FILENAME="${ESM2_CONTACT_REGRESSION_FILENAME}.debug"
fi

EM3DFOLD_WEIGHTS_SHA256="${EM3DFOLD_WEIGHTS_SHA256:-${DEFAULT_EM3DFOLD_WEIGHTS_SHA256}}"
RINALMO_WEIGHTS_SHA256="${RINALMO_WEIGHTS_SHA256:-${DEFAULT_RINALMO_WEIGHTS_SHA256}}"
ESM2_WEIGHTS_SHA256="${ESM2_WEIGHTS_SHA256:-${DEFAULT_ESM2_WEIGHTS_SHA256}}"
ESM2_CONTACT_REGRESSION_SHA256="${ESM2_CONTACT_REGRESSION_SHA256:-${DEFAULT_ESM2_CONTACT_REGRESSION_SHA256}}"

file_has_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -f "${path}" ]] || return 1
  actual="$(sha256sum "${path}")"
  actual="${actual%% *}"
  [[ "${actual}" == "${expected}" ]]
}

preserve_corrupt_file() {
  local path="$1"
  local backup="${path}.corrupt.$(date +%Y%m%d%H%M%S)"
  mv "${path}" "${backup}"
  echo "Preserved invalid file as: ${backup}" >&2
}

download_verified() {
  local label="$1"
  local url="$2"
  local destination="$3"
  local expected_sha256="$4"
  local partial="${destination}.part"

  if file_has_sha256 "${destination}" "${expected_sha256}"; then
    echo "${label}: checksum OK; skip download"
    return
  fi
  if [[ -f "${destination}" ]]; then
    preserve_corrupt_file "${destination}"
  fi

  echo "${label}: download (an incomplete .part file will be resumed)"
  run_step "Failed to download ${label}; rerun the same command to resume" \
    wget --continue --tries=5 --timeout=30 --waitretry=5 \
      "${url}" -O "${partial}"

  if ! file_has_sha256 "${partial}" "${expected_sha256}"; then
    preserve_corrupt_file "${partial}"
    fail "Checksum verification failed for ${label}"
  fi
  mv "${partial}" "${destination}"
  echo "${label}: checksum OK"
}

echo "Checking required commands"
require_cmd wget
require_cmd tar
require_cmd sha256sum

mkdir -p "${DOWNLOAD_ROOT}"
DOWNLOAD_ROOT="$(cd "${DOWNLOAD_ROOT}" && pwd)"
MODEL_WEIGHTS_DIR="${DOWNLOAD_ROOT}/weights"
LM_WEIGHTS_DIR="${DOWNLOAD_ROOT}/lm_weights"

mkdir -p "${MODEL_WEIGHTS_DIR}"
mkdir -p "${LM_WEIGHTS_DIR}"

echo "Download source: ${DOWNLOAD_SOURCE}"
if [[ "${DEBUG_DOWNLOAD}" == true ]]; then
  echo "Debug LM downloads: enabled (.debug URL suffix)"
fi

echo "1. Download EM3DFold pretrained weights"
download_verified \
  "EM3DFold pretrained weights" \
  "${EM3DFOLD_WEIGHTS_URL}" \
  "${MODEL_WEIGHTS_DIR}/weights.tgz" \
  "${EM3DFOLD_WEIGHTS_SHA256}"

WEIGHTS_EXTRACT_DIR="${MODEL_WEIGHTS_DIR}/weights"
WEIGHTS_EXTRACT_MARKER="${WEIGHTS_EXTRACT_DIR}/.weights.sha256"
weights_are_extracted() {
  [[ -f "${WEIGHTS_EXTRACT_DIR}/cpx/model_dual" ]] \
  && [[ -f "${WEIGHTS_EXTRACT_DIR}/protein/model_all_atom" ]] \
  && [[ -f "${WEIGHTS_EXTRACT_DIR}/protein/model_all_atom_no_lm" ]] \
  && [[ -f "${WEIGHTS_EXTRACT_DIR}/na/model_all_atom" ]] \
  && [[ -f "${WEIGHTS_EXTRACT_DIR}/na/model_all_atom_no_lm" ]] \
  && [[ -f "${WEIGHTS_EXTRACT_DIR}/na/model_na_aa_new" ]]
}

if weights_are_extracted \
  && { [[ ! -f "${WEIGHTS_EXTRACT_MARKER}" ]] \
    || [[ "$(<"${WEIGHTS_EXTRACT_MARKER}")" == "${EM3DFOLD_WEIGHTS_SHA256}" ]]; }; then
  printf '%s\n' "${EM3DFOLD_WEIGHTS_SHA256}" > "${WEIGHTS_EXTRACT_MARKER}"
  echo "EM3DFold pretrained weights: already extracted; skip extraction"
else
  run_step "Failed at step 1, unable to extract EM3DFold pretrained weights" \
    tar -zxf "${MODEL_WEIGHTS_DIR}/weights.tgz" -C "${MODEL_WEIGHTS_DIR}"
  printf '%s\n' "${EM3DFOLD_WEIGHTS_SHA256}" > "${WEIGHTS_EXTRACT_MARKER}"
fi

echo "2. Download RiNALMo pretrained weights"
download_verified \
  "RiNALMo pretrained weights" \
  "${RINALMO_WEIGHTS_URL}" \
  "${LM_WEIGHTS_DIR}/${RINALMO_WEIGHTS_FILENAME}" \
  "${RINALMO_WEIGHTS_SHA256}"

echo "3. Download ESM2 pretrained weights"
download_verified \
  "ESM2 pretrained weights" \
  "${ESM2_WEIGHTS_URL}" \
  "${LM_WEIGHTS_DIR}/${ESM2_WEIGHTS_FILENAME}" \
  "${ESM2_WEIGHTS_SHA256}"
download_verified \
  "ESM2 contact regression weights" \
  "${ESM2_CONTACT_REGRESSION_URL}" \
  "${LM_WEIGHTS_DIR}/${ESM2_CONTACT_REGRESSION_FILENAME}" \
  "${ESM2_CONTACT_REGRESSION_SHA256}"

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
