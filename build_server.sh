#!/usr/bin/env bash
# Build the FoundationStereo depth-server environment.
#
# Uses torch 2.7.1+cu128 (NOT the torch 2.4.1 pinned in environment.yml, which has
# no sm_120/Blackwell kernels and so cannot run on the RTX 5090).
#
# Deliberately skipped vs environment.yml:
#   - flash-attn : never imported anywhere in the repo (readme mentions it only).
#   - xformers   : pinned to 0.0.28.post1 == torch 2.4.1; dinov2 guards its import.
#   - imgaug / albumentations / jupyterlab / nodejs : training-time only, not on the
#     server or run_demo.py import path.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] pixi install (python 3.11) ==="
pixi install

echo "=== [2/4] torch 2.7.1+cu128 (Blackwell-capable) ==="
pixi run pip install --no-cache-dir \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

echo "=== [3/4] runtime deps (server + run_demo import path) ==="
pixi run pip install --no-cache-dir \
  timm omegaconf opencv-contrib-python open3d pandas einops \
  huggingface_hub imageio scipy scikit-image trimesh "ruamel.yaml" \
  transformations gdown \
  fastapi "uvicorn[standard]" python-multipart

echo "=== [4/4] download weights (Google Drive -> pretrained_models/23-51-11) ==="
if [ -f pretrained_models/23-51-11/model_best_bp2.pth ]; then
  echo "weights already present, skipping download"
else
  mkdir -p pretrained_models
  # The readme's weights folder; holds both 23-51-11 (Vit-large, the default the
  # server loads) and 11-33-40 (Vit-small). gdown pulls the subfolders as-is.
  pixi run gdown --folder --continue -O pretrained_models \
    https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
fi

echo "=== verify: torch sees the 5090 + FoundationStereo builds + weights load ==="
pixi run python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch.cuda.get_arch_list())
assert torch.cuda.is_available(), 'CUDA not available'
import os
ck = 'pretrained_models/23-51-11/model_best_bp2.pth'
if os.path.isfile(ck):
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    cfg = OmegaConf.load('pretrained_models/23-51-11/cfg.yaml')
    if 'vit_size' not in cfg: cfg['vit_size'] = 'vitl'
    m = FoundationStereo(OmegaConf.create(dict(cfg)))
    m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    print('FoundationStereo weights loaded OK')
else:
    print('WARNING: no checkpoint at', ck, '-- server will report status=unconfigured')
"
echo "BUILD_OK"
