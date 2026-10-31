#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
export MACOSX_DEPLOYMENT_TARGET=13.0
case "$(uname -m)" in
  arm64) target=aarch64-apple-darwin ;;
  *) target=x86_64-apple-darwin ;;
esac
uv sync --frozen --python 3.12 --python-platform "$target" --extra dev --extra build
uv pip install --python .venv/bin/python cmake==3.31.6 ninja==1.11.1.3 wheel==0.45.1
uv run --no-sync python tools/build_onnxruntime_macos.py --jobs 2 --install
uv run --no-sync python tools/build_desktop.py
