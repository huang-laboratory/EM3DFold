#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_EXE="${MAMBA_EXE:-mamba}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/download_with_mamba.sh DOWNLOAD_ROOT

Optional environment variables:
  MAMBA_EXE             Mamba executable (default: mamba from PATH)
  EM3DFOLD_ENV_NAME     Environment name (default: em3dfold)
  EM3DFOLD_ENV_PREFIX   Set EM_WEIGHTS_DIR in this explicit prefix instead
                        of a named environment
EOF
  exit 0
fi

if [[ "${MAMBA_EXE}" == */* ]]; then
  [[ -x "${MAMBA_EXE}" ]] || {
    echo "Mamba executable is not executable: ${MAMBA_EXE}" >&2
    exit 1
  }
else
  MAMBA_EXE="$(command -v "${MAMBA_EXE}")" || {
    echo "Required command not found in PATH: ${MAMBA_EXE}" >&2
    exit 1
  }
fi

export EM3DFOLD_ENV_MANAGER="${MAMBA_EXE}"
exec bash "${SCRIPT_DIR}/download.sh" "$@"
