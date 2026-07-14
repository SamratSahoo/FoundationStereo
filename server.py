"""FoundationStereo depth HTTP server (for the tiptop perception pipeline).

Exposes the API tiptop expects (see tiptop/tiptop/perception/foundation_stereo.py):

  POST /infer   (multipart: left_image, right_image PNGs + form fields
                 fx, fy, cx, cy, baseline, scale, hiera, valid_iters)
        -> NPZ bytes with key "depth" (float32 HxW, metres)
  GET  /health  -> {"status": "healthy"}  once the model is loaded,
                   else {"status": "unconfigured"} (no checkpoint found)

NOTE: For the droid-sim tiptop eval this server is NOT required -- the tiptop
websocket server runs perception with depth_estimator=None and uses the depth the
simulator provides, so FoundationStereo is never called. It is provided so the
full perception stack can be stood up for real-robot use. It needs the
FoundationStereo conda/pixi environment (see environment.yml: dinov2,
depth_anything, flash-attn, ...) and the gated pretrained weights placed under
pretrained_models/<run>/model_best_bp2.pth (+ cfg.yaml). Until those exist,
/health reports "unconfigured" and full_eval (--launch-fs) treats it as a soft
dependency and continues.

Run from the FoundationStereo repo root:
    python server.py --port 8124 --ckpt-dir pretrained_models/<run>/model_best_bp2.pth
"""

import argparse
import io
import logging
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("foundation_stereo_server")

_MODEL = None
_ARGS = None


def load_model(ckpt_dir: str) -> bool:
    """Load FoundationStereo from a checkpoint dir. Returns False if weights are absent."""
    global _MODEL, _ARGS
    if not os.path.isfile(ckpt_dir):
        logger.warning(f"FoundationStereo checkpoint not found at {ckpt_dir}; serving in 'unconfigured' mode")
        return False
    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    args = OmegaConf.create(dict(cfg))
    model = FoundationStereo(args)
    # weights_only=False: torch>=2.6 flipped this default to True, which rejects the
    # numpy scalars pickled in the released checkpoints.
    ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    _MODEL = model.cuda().eval()
    _ARGS = args
    logger.info(f"Loaded FoundationStereo from {ckpt_dir}")
    return True


def _infer_depth(left_rgb, right_rgb, fx, baseline, scale, hiera, valid_iters):
    import cv2
    import torch
    from core.utils.utils import InputPadder

    if scale != 1.0:
        left_rgb = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None)
        right_rgb = cv2.resize(right_rgb, fx=scale, fy=scale, dsize=None)
    h, w = left_rgb.shape[:2]
    img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        if hiera:
            disp = _MODEL.run_hierachical(img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5)
        else:
            disp = _MODEL.forward(img0, img1, iters=valid_iters, test_mode=True)
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)
    # disparity -> metric depth (fx already scaled with the image)
    depth = (fx * scale) * baseline / np.clip(disp, 1e-6, None)
    return depth.astype(np.float32)


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy" if _MODEL is not None else "unconfigured"}


@app.post("/infer")
async def infer(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    baseline: float = Form(...),
    scale: float = Form(1.0),
    hiera: int = Form(0),
    valid_iters: int = Form(32),
):
    if _MODEL is None:
        return Response(content="FoundationStereo model not loaded (no weights)", status_code=503)
    import cv2

    left = cv2.cvtColor(cv2.imdecode(np.frombuffer(await left_image.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imdecode(np.frombuffer(await right_image.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = _infer_depth(left, right, fx, baseline, scale, bool(hiera), int(valid_iters))
    buf = io.BytesIO()
    np.savez(buf, depth=depth)
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


def main():
    ap = argparse.ArgumentParser(description="FoundationStereo depth server")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--ckpt-dir", type=str, default="pretrained_models/23-51-11/model_best_bp2.pth")
    args = ap.parse_args()
    load_model(args.ckpt_dir)
    logger.info(f"Serving FoundationStereo on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
