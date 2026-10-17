"""Content-hash cache for API calls. Never pay twice for the same (image, settings, model) call."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, db_path: str | Path = "results/cache.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                image_hash TEXT,
                model TEXT,
                settings_json TEXT,
                response_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _hash_image(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def _make_key(image_hash: str, model: str, settings: dict[str, Any]) -> str:
        settings_str = json.dumps(settings, sort_keys=True)
        raw = f"{image_hash}|{model}|{settings_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, image_bytes: bytes, model: str, settings: dict[str, Any]) -> dict | None:
        image_hash = self._hash_image(image_bytes)
        key = self._make_key(image_hash, model, settings)
        row = self._conn.execute(
            "SELECT response_json FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, image_bytes: bytes, model: str, settings: dict[str, Any], response: dict
    ) -> None:
        image_hash = self._hash_image(image_bytes)
        key = self._make_key(image_hash, model, settings)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, image_hash, model, settings_json, response_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, image_hash, model, json.dumps(settings, sort_keys=True), json.dumps(response)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
