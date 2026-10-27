#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
export MACOSX_DEPLOYMENT_TARGET=13.0
case "$(uname -m)" in
  arm64) target=aarch64-apple-darwin ;;
  *) target=x86_64-apple-darwin ;;
esac
uv sync --frozen --python-platform "$target" --extra dev --extra build
uv run --no-sync python tools/build_desktop.py
