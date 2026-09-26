#!/usr/bin/env bash
# Build the FoundationStereo depth server inside this workspace's pixi environment: `pixi run setup`.
#
# What build_server.sh does, minus its own `pixi install`/`pixi run` calls, so it can run as a pixi task
# (tandem builds this server as a runtime and runs the task). The weights are their own task
# (download-weights).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install --no-cache-dir \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
pip install --no-cache-dir \
  timm omegaconf opencv-contrib-python open3d pandas einops \
  huggingface_hub imageio scipy scikit-image trimesh "ruamel.yaml" \
  transformations gdown \
  fastapi "uvicorn[standard]" python-multipart
echo "SETUP_OK"
