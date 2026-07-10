"""One-time export of the FastReID VeRi checkpoint -> ONNX (run inside the Docker env
built by export_in_docker.sh; NOT part of the pipeline venv).

Builds the exact FastReID architecture from configs/VeRi/sbs_R50-ibn.yml, loads the
checkpoint, and does a plain torch.onnx.export (skipping fast-reid's fragile
onnx-simplify/optimize steps). The traced graph BAKES IN pixel_mean/std normalization,
so the resulting ONNX expects raw 0-255 RGB (B,3,256,256) and outputs the 2048-d
BNNeck feature.
"""

import sys

import torch
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

CONFIG = "fast-reid/configs/VeRi/sbs_R50-ibn.yml"
WEIGHTS = "/w/veri_sbs_R50-ibn.pth"
OUT = "/w/veri_reid.onnx"


def main() -> int:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 575  # VeRi train identities (matches classifier shape)
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
