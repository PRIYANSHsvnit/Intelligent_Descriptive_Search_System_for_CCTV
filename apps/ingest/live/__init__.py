"""LIVE tier — streaming ingest driver.

A SECOND driver over the frozen batch stages in ``pipeline/`` (which stays the "not-live"
build). Where ``run_ingest.py`` is stage-major (load one model, sweep the whole clip, write
files, bulk-insert), the live tier keeps all models CO-RESIDENT (~2.7 GB, measured — see the
live-tier-vram memory) and finalizes each tracklet the moment its track ends: best-K crops →
SigLIP embed → color → INSERT, streaming into the DB. Vehicle plates are read by async CPU OCR
off the GPU hot path.

Everything here reuses ``pipeline/`` by IMPORT only — no edits to the batch code, so the
existing demo build is never at risk. reid (DINOv3, offline/s02 story) and the VLM
(offline tier) are deliberately absent from the live path.
"""
