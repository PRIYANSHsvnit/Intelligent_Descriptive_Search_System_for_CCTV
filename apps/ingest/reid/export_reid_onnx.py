"""One-time export of a FastReID vehicle checkpoint -> ONNX (run inside the Docker env
built by export_in_docker.sh; NOT part of the pipeline venv).

Builds the FastReID architecture from a config, loads the checkpoint, and does a plain
torch.onnx.export (skipping fast-reid's fragile onnx-simplify/optimize). The traced graph
BAKES IN pixel_mean/std normalization, so the ONNX expects raw 0-255 RGB (B,3,H,W) and
outputs the 2048-d BNNeck feature.

Parametrized via env (defaults = the original VeRi sbs export, backward compatible):
  REID_CFG     config path (relative to /w)   default fast-reid/configs/VeRi/sbs_R50-ibn.yml
  REID_WEIGHTS checkpoint path                 default /w/veri_sbs_R50-ibn.pth
  REID_OUT     output onnx path               default /w/veri_reid.onnx
NUM_CLASSES is auto-detected from the checkpoint's classifier weight (the head is unused
for feature export, but the shape must match to load cleanly).
"""

import os
import sys

import torch
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

CONFIG = os.environ.get("REID_CFG", "fast-reid/configs/VeRi/sbs_R50-ibn.yml")
WEIGHTS = os.environ.get("REID_WEIGHTS", "/w/veri_sbs_R50-ibn.pth")
OUT = os.environ.get("REID_OUT", "/w/veri_reid.onnx")


def _detect_num_classes(weights: str, default: int = 575) -> int:
    ck = torch.load(weights, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    for k, v in sd.items():
        if "classifier" in k and hasattr(v, "ndim") and v.ndim >= 1:
            print(f"detected classifier {k} -> num_classes={v.shape[0]}")
            return int(v.shape[0])
    print(f"no classifier key found; using default num_classes={default}")
    return default


def main() -> int:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = _detect_num_classes(WEIGHTS)
    if cfg.MODEL.HEADS.POOL_LAYER == "FastGlobalAvgPool":
        cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
    cfg.freeze()

    model = build_model(cfg)
    Checkpointer(model).load(WEIGHTS)
    if hasattr(model.backbone, "deploy"):
        model.backbone.deploy(True)
    model.eval()

    dummy = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
    with torch.no_grad():
        out = model(dummy)
    print("feature shape:", tuple(out.shape))  # expect (1, 2048)

    torch.onnx.export(
        model, dummy, OUT,
        input_names=["input"], output_names=["feat"],
        dynamic_axes={"input": {0: "batch"}, "feat": {0: "batch"}},
        opset_version=11, do_constant_folding=True,
    )
    print("exported ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
