"""Co-resident model pool for the live tier: both YOLO detectors + SigLIP2 loaded ONCE
and kept in VRAM, reused across every frame and camera. reid is NOT here (offline/s02);
the VLM is offline-tier only.

Reuses the frozen batch stages by import: the SigLIP load + get_image_features mirror
``embed_siglip``; the color text matrix + margin rule are ``color_siglip`` logic with the
model already resident (no re-load, no per-camera reload).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

from pipeline import profiles
from pipeline.cfg import COLOR_MIN_MARGIN, COLOR_VOCAB, SEMANTIC_DIM, SIGLIP_MODEL
from pipeline.color_siglip import _color_text_matrix


class ModelPool:
    """Loads and holds the live-tier GPU model set. Call ``profiles.use(...)`` first."""

    def __init__(self, device: int = 0):
        prof = profiles.active()
        self.prof = prof
        self.pdet = prof.person_detector
        self.device = device
        use_cuda = torch.cuda.is_available()
        self.dev = f"cuda:{device}" if use_cuda else "cpu"
        self.dtype = torch.float16 if use_cuda else torch.float32

        # detectors — co-resident, each keeps its own persistent BoT-SORT state
        self.veh = YOLO(prof.yolo_model)
        self.per = YOLO(self.pdet.weights) if self.pdet is not None else None

        # SigLIP: image tower (embed) + text tower (used once to build the color matrix)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        self.siglip = AutoModel.from_pretrained(SIGLIP_MODEL, dtype=self.dtype).to(self.dev).eval()
        self.color_text = _color_text_matrix(self.siglip, self.processor, self.dev)  # (C, D)

    @torch.no_grad()
    def embed_views(self, crops_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return (normalized mean, normalized per-crop vectors) for BGR crops."""
        if not crops_bgr:
            return (np.zeros(SEMANTIC_DIM, dtype=np.float32),
                    np.zeros((0, SEMANTIC_DIM), dtype=np.float32))
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        pv = self.processor(images=rgb, return_tensors="pt").pixel_values.to(self.dev, self.dtype)
        feats = self.siglip.get_image_features(pixel_values=pv)
        if not isinstance(feats, torch.Tensor):  # transformers 5.x output object
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()
        v = feats.mean(axis=0)
        mean = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        return mean, feats.astype(np.float32)

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Compatibility wrapper for callers that only need the normalized mean."""
        return self.embed_views(crops_bgr)[0]

    def color_name(self, sem: np.ndarray) -> tuple[str | None, float | None]:
        """Zero-shot color = argmax_c cos(sem, text_c); None if empty or below the
        commit margin (mirrors color_siglip.run's per-tracklet rule; caller gates to vehicles)."""
        if np.linalg.norm(sem) < 1e-6:
            return None, None
        sims = self.color_text @ sem  # (C,), both L2-normed
        order = np.argsort(-sims)
        top1 = float(sims[order[0]])
        top2 = float(sims[order[1]]) if len(COLOR_VOCAB) > 1 else -np.inf
        if top1 - top2 < COLOR_MIN_MARGIN:
            return None, None
        return COLOR_VOCAB[int(order[0])], round(top1, 4)

    def reset_trackers(self) -> None:
        """Fresh BoT-SORT id namespaces for the next camera (models stay resident)."""
        for m in (self.veh, self.per):
            if m is None:
                continue
            pred = getattr(m, "predictor", None)
            for tr in getattr(pred, "trackers", None) or []:
                if hasattr(tr, "reset"):
                    tr.reset()
