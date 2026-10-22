#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_EXE="${MAMBA_EXE:-mamba}"
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

resolve_executable() {
  local executable="$1"
  if [[ "${executable}" == */* ]]; then
    [[ -x "${executable}" ]] || fail "Mamba executable is not executable: ${executable}"
    printf '%s\n' "${executable}"
    return
  fi
  command -v "${executable}" || fail "Required command not found in PATH: ${executable}"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/install_with_mamba.sh

Optional environment variables:
  MAMBA_EXE             Mamba executable (default: mamba from PATH)
  EM3DFOLD_ENV_NAME     Environment name (default: em3dfold)
  EM3DFOLD_ENV_PREFIX   Create and install into this explicit prefix instead
                        of a named environment
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  usage >&2
  exit 1
fi

MAMBA_EXE="$(resolve_executable "${MAMBA_EXE}")"

if [[ -n "${ENV_PREFIX}" ]]; then
  ENV_PREFIX="$(realpath -m "${ENV_PREFIX}")"
  create_target=(--prefix "${ENV_PREFIX}")
  run_target=(--prefix "${ENV_PREFIX}")
  target_description="prefix ${ENV_PREFIX}"
else
  create_target=(--name "${ENV_NAME}")
  run_target=(--name "${ENV_NAME}")
  target_description="environment ${ENV_NAME}"
fi

echo "1 Create Mamba ${target_description}"
run_step "Failed at step 1, unable to create ${target_description}" \
  "${MAMBA_EXE}" env create --yes --file "${ROOT_DIR}/env.yml" "${create_target[@]}"

echo "2 Install flash attention"
run_step "Failed at step 2, unable to install flash attention" \
  "${MAMBA_EXE}" run "${run_target[@]}" --cwd "${ROOT_DIR}/flash_attn_whl" \
  bash install_flash_attn.sh
echo "Done install base env"

echo "3 Install EM3DFold"
run_step "Failed at step 3, unable to install EM3DFold" \
  "${MAMBA_EXE}" run "${run_target[@]}" python -m pip install "${ROOT_DIR}"
echo "Done install EM3DFold"

echo "4 Check installation"
run_step "Failed at step 4, python -m em3dfold --help did not run successfully" \
  "${MAMBA_EXE}" run "${run_target[@]}" python -m em3dfold --help
echo "Done check installation"

if [[ -n "${ENV_PREFIX}" ]]; then
  echo "Run EM3DFold with:"
  echo "  ${MAMBA_EXE} run --prefix ${ENV_PREFIX} em3dfold --help"
else
  echo "Activate the environment with:"
  echo "  eval \"\$(\"${MAMBA_EXE}\" shell hook -s bash)\""
  echo "  mamba activate ${ENV_NAME}"
fi
