"""Postgres + pgvector connection helpers."""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

DEFAULT_DSN = "postgresql://cctv:cctv@localhost:5432/cctv"


def dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def connect() -> psycopg.Connection:
    conn = psycopg.connect(dsn())
    register_vector(conn)
    return conn
