"""Offline ingest pipeline (Phase 1): detect → track → crop → attributes → embed → store.

Passes are decoupled and re-runnable; crops persist to disk between them so only one
model sits on the GPU at a time (see plan.md "Memory: why 6 GB is plenty").
"""
