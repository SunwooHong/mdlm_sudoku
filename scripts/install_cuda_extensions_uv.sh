#!/usr/bin/env bash
# causal-conv1d, mamba-ssm, flash-attn 소스 빌드 설치
# 전제: uv sync로 .venv 생성 완료 + CUDA toolkit(nvcc) 사용 가능
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Run first: cd $ROOT && uv sync" >&2
  exit 1
fi

# Try to expose nvcc on clusters that use environment modules.
if ! command -v nvcc >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.6 >/dev/null 2>&1 || module load cuda/12.2 >/dev/null 2>&1 || true
  fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
nvcc not found.
Load a CUDA toolkit module first (example):
  module load cuda/12.6
Then rerun:
  ./scripts/install_cuda_extensions_uv.sh
EOF
  exit 2
fi

UV_PROJECT="$ROOT" uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation \
  "causal-conv1d==1.1.3.post1" \
  "mamba-ssm==1.1.4" \
  "flash-attn==2.5.6"
