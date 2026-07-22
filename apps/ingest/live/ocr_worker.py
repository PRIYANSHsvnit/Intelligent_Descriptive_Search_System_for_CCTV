"""Async CPU plate OCR for the live tier.

Plate OCR is CPU-only (RapidOCR / ONNX) so it never competes for the ~2.7 GB of VRAM the
GPU hot path uses. But it's slow (~10 crops/s), so it runs OFF the frame loop: vehicle
tracklets are enqueued at finalize, a small thread pool reads them, and the row's plate_*
fields are UPDATEd a beat later (eventually-consistent — fine for the live/provisional tier).
This keeps a burst of vehicles from stalling detection AND keeps plate work out of the
GPU-path fps number.

Reuses the frozen ``pipeline/plate.py`` constraint stack by import (detect → crop → read →
repair → resolve). ONNX InferenceSession.run is thread-safe, so the worker threads share one
detector/OCR pair — but here we build it OURSELVES with capped ONNX thread pools rather than
calling ``plate._models()``: RapidOCR's config defaults every session to intra_op=-1 (all
cores), so N uncapped workers × 3 sessions each stampede all cores and starve the GPU-feeding
threads. We cap each session to a couple of threads; plate.py stays untouched (frozen).
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import onnxruntime as ort

from pipeline import plate as P
from .writer import DBWriter


def _capped_models(intra_threads: int):
    """detector + OCR identical to plate._models(), but with ONNX intra-op threads capped so
    the OCR pool can't grab every core (the live-tier oversubscription fix)."""
    from open_image_models import LicensePlateDetector
    from rapidocr_onnxruntime import RapidOCR
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = 1
    detector = LicensePlateDetector(
        detection_model=P.DETECTOR_MODEL, conf_thresh=P.DETECTOR_CONF, sess_options=so)
    # RapidOCR copies the Global intra/inter values onto Det/Cls/Rec (update_global_to_module),
    # clobbering any per-section kwargs — so set the GLOBAL (non-prefixed) knob, which then
    # propagates to all three ONNX sessions.
    ocr = RapidOCR(intra_op_num_threads=intra_threads, inter_op_num_threads=1)
    return detector, ocr


class AsyncOCR:
    def __init__(self, workers: int = 2, intra_threads: int = 2) -> None:
        self.q: "queue.Queue" = queue.Queue()
        self.workers = workers
        self.intra_threads = intra_threads
        self.threads: list[threading.Thread] = []
        self.enqueued = 0
        self.done = 0
        self.asserted = 0            # tracklets that yielded a validated plate_text
        self.writer = DBWriter()     # own connection; serialized by _lock below
        self._lock = threading.Lock()
        self._models = None          # capped detector+OCR, shared across workers

    def start(self) -> None:
        # build the capped detector+OCR once, before workers spin (thread-safe to share)
        self._models = _capped_models(self.intra_threads)
        for _ in range(self.workers):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self.threads.append(t)

    def submit(self, tracklet_id: str, frame_span: int, crops_bgr: list[np.ndarray]) -> None:
        self.enqueued += 1
        # copy: the engine may recycle/free these crops after finalize
        self.q.put((tracklet_id, frame_span, [c.copy() for c in crops_bgr]))

    def _loop(self) -> None:
        detector, ocr = self._models
        while True:
            item = self.q.get()
            if item is None:          # stop sentinel
                self.q.task_done()
                return
            tid, span, crops = item
            try:
                text, conf, raw = self._read(detector, ocr, span, crops)
                if text or raw:
                    with self._lock:
                        self.writer.update_plate(tid, text, conf, raw)
                    if text:
                        self.asserted += 1
            finally:
                self.done += 1
                self.q.task_done()

    def _read(self, detector, ocr, span: int, crops: list[np.ndarray]):
        """Per-tracklet plate read over its best crops — mirrors plate.run's per-crop loop."""
        reads = []
        for img in crops:
            box = P._detect_plate(detector, img)
            if box is None:
                continue
            pc = P._plate_crop(img, box)
            if pc is None:
                continue
            raw, conf = P._read_plate(ocr, pc)
            if not raw:
                continue
            canon, repaired = P._repair(raw)
            reads.append({"raw": raw, "conf": conf, "canon": canon, "repaired": repaired})
        can_agree = span >= P.INDEP_FRAME_GAP
        return P._resolve(reads, can_agree)

    def drain(self) -> None:
        """Block until every submitted tracklet has been processed."""
        self.q.join()

    def stop(self) -> None:
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()
        self.writer.close()
