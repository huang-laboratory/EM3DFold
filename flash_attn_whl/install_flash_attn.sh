#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() {
  echo "$*" >&2
  exit 1
}

run_step() {
  local msg="$1"
  shift
  "$@" || fail "$msg"
}

flash_attn_version="$(python -c 'import importlib.metadata as m; print(m.version("flash-attn"))' 2>/dev/null || true)"
if [[ "${flash_attn_version}" == "2.3.2" ]] && python -c 'import flash_attn' >/dev/null 2>&1; then
  echo "Flash Attention 2.3.2 is already installed; skip download"
  exit 0
fi

cxx11_abi="$(python -c "import torch; print(torch.compiled_with_cxx11_abi())")"
cxx11_abi=${cxx11_abi^^}
wheel_name="flash_attn-2.3.2+cu118torch2.1cxx11abi${cxx11_abi}-cp310-cp310-linux_x86_64.whl"
wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/${wheel_name}"
wheel_path="${SCRIPT_DIR}/${wheel_name}"
partial_path="${wheel_path}.part"

case "${cxx11_abi}" in
  FALSE) expected_size=57075024 ;;
  TRUE) expected_size=57076786 ;;
  *) fail "Unsupported PyTorch CXX11 ABI value: ${cxx11_abi}" ;;
esac

wheel_is_valid() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]] || return 1
  python -m zipfile -t "${path}" >/dev/null 2>&1
}

if wheel_is_valid "${wheel_path}"; then
  echo "Use cached Flash Attention wheel: ${wheel_path}"
else
  if [[ -f "${wheel_path}" ]]; then
    mv "${wheel_path}" "${wheel_path}.corrupt.$(date +%Y%m%d%H%M%S)"
  fi
  echo "Download Flash Attention wheel (an incomplete .part file will be resumed)"
  run_step "Failed to download Flash Attention wheel; rerun the same command to resume" \
    wget --continue --tries=5 --timeout=30 --waitretry=5 \
      "${wheel_url}" -O "${partial_path}"
  if ! wheel_is_valid "${partial_path}"; then
    mv "${partial_path}" "${partial_path}.corrupt.$(date +%Y%m%d%H%M%S)"
    fail "Downloaded Flash Attention wheel is incomplete or invalid"
  fi
  mv "${partial_path}" "${wheel_path}"
fi

echo "Install ${wheel_name}"
run_step "Failed to install flash attention wheel" \
  python -m pip install "${wheel_path}"

run_step "Flash Attention import check failed after installation" \
  python -c 'import flash_attn'
